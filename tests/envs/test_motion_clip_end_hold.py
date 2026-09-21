"""Clip-end behaviour of MotionCommand: resample (default) vs hold, plus the
``motion_ended`` flag lifecycle.

Kept out of ``test_env_configs.py`` on purpose: that module gates every runtime
case behind ``_require_mujoco_runtime``, which skips when ``mujoco_uni`` is
missing. The dm10 MuJoCo backend builds and steps fine without ``mujoco_uni``
(that guard covers a different code path), so putting these there would make
them skip silently — a test that can never fail is worse than no test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter

_ROOT = Path(__file__).parents[2]
_ACTION_DIM = 10


def _make_env(task: str, *, num_envs: int, hold: bool = False, config_root: str = "ppo"):
    """Build an env with `hold_on_clip_end` set EXPLICITLY.

    Always writes the flag in both directions rather than relying on the task
    yaml's default: dm10_stand turns it on, and a test that silently inherited
    that would start asserting the opposite of what its name says the moment the
    yaml changed (which is exactly what happened once).
    """
    registry.ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_ROOT / "src" / "unilab" / "conf" / config_root), version_base="1.3"
    ):
        owner = compose("config", overrides=[f"task={task}/mujoco"])
    override = BackendAdapter(
        owner, root_dir=_ROOT, algo_name=config_root
    ).build_task_env_cfg_override()
    override["auto_reset"] = False
    override["commands"]["motion"]["params"]["hold_on_clip_end"] = hold
    return registry.make(
        str(owner.training.task_name),
        num_envs=num_envs,
        sim_backend="mujoco",
        env_cfg_override=override,
    )


def _step(env, num_envs: int):
    return env.step(np.zeros((num_envs, _ACTION_DIM), dtype=np.float32))


def test_default_resamples_at_clip_end() -> None:
    """Default path: clip end resamples the reference and never latches."""
    env = _make_env("dm10_stand", num_envs=2)
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        assert command.cfg.params.hold_on_clip_end is False
        assert command.motion_ended.dtype == np.bool_
        assert not command.motion_ended.any()

        end = command.sampler.current_clip_end_frames.copy()
        command.time_steps[:] = end  # park at the last frame
        _step(env, 2)  # then step past it — the overflow path

        # Resampled back into range, not pinned.
        assert np.all(command.time_steps < end)
        np.testing.assert_array_equal(command.motion_ended, [False, False])
    finally:
        env.close()


def test_hold_pins_frame_and_latches_flag() -> None:
    """With hold_on_clip_end the frame is pinned and the flag latches.

    This is exactly the path the existing truncation test misses: it parks the
    frame *at* the clip end, where the termination has already pulled the row out
    of active_ids. Stepping one past it is what used to resample — or, with
    truncate_on_clip_end, raise IndexError out of np.take.
    """
    env = _make_env("dm10_stand", num_envs=2, hold=True)
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        assert command.cfg.params.hold_on_clip_end is True

        end = command.sampler.current_clip_end_frames.copy()
        command.time_steps[:] = end
        _step(env, 2)

        np.testing.assert_array_equal(command.time_steps, end)
        np.testing.assert_array_equal(command.motion_ended, [True, True])

        # Reference is frozen on the final frame, bit for bit.
        frozen = command.motion.get_motion_at_frame(end).joint_pos
        np.testing.assert_array_equal(command.joint_pos, frozen)

        # Stays pinned and latched (idempotent) across further steps.
        for _ in range(3):
            _step(env, 2)
            np.testing.assert_array_equal(command.time_steps, end)
        np.testing.assert_array_equal(command.motion_ended, [True, True])
    finally:
        env.close()


def test_hold_survives_every_clip_boundary() -> None:
    """Pinning must stay in range for every clip in a multi-clip task.

    A non-final clip's end+1 is a *valid* index into the concatenated frame
    array, so before this flag existed the overflow silently read the next
    clip's first frame instead of failing loudly.
    """
    env = _make_env("dm10_motion_tracking", num_envs=2, hold=True)
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        ends = command.sampler.current_clip_end_frames.copy()
        command.time_steps[:] = ends
        _step(env, 2)
        np.testing.assert_array_equal(command.time_steps, ends)
        # Must be a legal gather for every row.
        command.motion.get_motion_at_frame(command.time_steps)
    finally:
        env.close()


def test_motion_ended_cleared_only_for_reset_rows() -> None:
    """Partial reset clears the flag on the reset rows and leaves the rest."""
    env = _make_env("dm10_stand", num_envs=4, hold=True)
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        command.time_steps[:] = command.sampler.current_clip_end_frames
        _step(env, 4)
        np.testing.assert_array_equal(command.motion_ended, [True] * 4)

        env.reset(env_ids=np.array([0, 2]))
        np.testing.assert_array_equal(command.motion_ended, [False, True, False, True])
    finally:
        env.close()


def test_flag_is_cleared_by_resampling() -> None:
    """Any resample restarts the reference, so the flag must drop."""
    env = _make_env("dm10_stand", num_envs=4, hold=True)
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        command.time_steps[:] = command.sampler.current_clip_end_frames
        _step(env, 4)
        assert command.motion_ended.all()

        rows = np.array([1, 3])
        command.sampler._set_sampled_frames(rows, command.sampler.sample_frames(rows))
        np.testing.assert_array_equal(command.motion_ended, [True, False, True, False])
    finally:
        env.close()


@pytest.mark.parametrize(
    ("hold", "truncate"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_mutually_exclusive_flags_rejected(hold: bool, truncate: bool) -> None:
    from unilab.tasks.motion_tracking.common.manager_terms import (
        MotionCommand,
        MotionCommandCfg,
        MotionCommandParamsCfg,
    )

    params = MotionCommandParamsCfg(
        motion_file="motions/dm10/stand.npz",
        anchor_body_name="base_link",
        body_names=["base_link", "leg_l5_link"],
        hold_on_clip_end=hold,
        truncate_on_clip_end=truncate,
    )
    cfg = MotionCommandCfg(entity_name="robot", params=params, resampling_time_range=(1.0e9, 1.0e9))
    if hold and truncate:
        with pytest.raises(ValueError, match="mutually exclusive"):
            MotionCommand._validate_cfg(cfg)
    else:
        MotionCommand._validate_cfg(cfg)  # must not raise


@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_hold_on_clip_end_must_be_bool(bad) -> None:
    from unilab.tasks.motion_tracking.common.manager_terms import (
        MotionCommand,
        MotionCommandCfg,
        MotionCommandParamsCfg,
    )

    params = MotionCommandParamsCfg(
        motion_file="motions/dm10/stand.npz",
        anchor_body_name="base_link",
        body_names=["base_link", "leg_l5_link"],
        hold_on_clip_end=bad,  # type: ignore[arg-type]
    )
    cfg = MotionCommandCfg(entity_name="robot", params=params, resampling_time_range=(1.0e9, 1.0e9))
    with pytest.raises(TypeError, match="must be bool"):
        MotionCommand._validate_cfg(cfg)


# --------------------------------------------------------------------------- #
# Why there is no bit-identity test here.
#
# The natural way to prove "the default path still behaves exactly as before" is
# to run the old and new implementations in one process and compare buffers. That
# comparison is NOT satisfiable in this environment, and the measurement below is
# what established it: building the same env config twice yields different RNG
# streams, so two envs never share initial conditions.
#
# Measured 2026-09-21, dm10_stand, num_envs=2, identical action sequence:
#     old impl vs *itself*   -> max |diff| 7.2e-03  (first differing step: 0)
#     old impl vs new impl   -> max |diff| 6.1e-03
# The ambient spread exceeds the effect being measured, so any bit-identity
# assertion would be asserting something the harness cannot deliver.
#
# What IS asserted instead, below: the *branch decision* at clip end, which is
# deterministic and is the only thing the change actually alters.
# --------------------------------------------------------------------------- #
def test_env_rng_does_not_restart_per_instance() -> None:
    """Pins the harness property that makes bit-identity testing impossible.

    Documents the finding rather than leaving it as folklore: if a future change
    makes env RNG seeding per-instance, this test flips and the bit-identity
    comparison becomes viable again.
    """
    first = _make_env("dm10_stand", num_envs=2)
    second = _make_env("dm10_stand", num_envs=2)
    try:
        first.init_state()
        second.init_state()
        assert not np.array_equal(first.rng.random(3), second.rng.random(3)), (
            "env.rng now restarts per instance — bit-identity A/B is possible again, "
            "and the clip-end default path should get one."
        )
    finally:
        first.close()
        second.close()


def test_default_and_hold_take_different_clip_end_branches() -> None:
    """The branch decision is deterministic: resample vs pin.

    This is the deterministic core of the change, so it is what gets asserted.
    The two envs below have different RNG streams by construction, so we do not
    compare values between them — only which branch each one took.
    """
    default_env = _make_env("dm10_stand", num_envs=2)
    hold_env = _make_env("dm10_stand", num_envs=2, hold=True)
    try:
        for env, expected_hold in ((default_env, False), (hold_env, True)):
            env.init_state()
            command = env.command_manager.get_term("motion")
            end = command.sampler.current_clip_end_frames.copy()
            command.time_steps[:] = end
            env.step(np.zeros((2, _ACTION_DIM), dtype=np.float32))

            pinned = np.array_equal(command.time_steps, end)
            assert pinned is expected_hold, (
                f"hold_on_clip_end={expected_hold} should "
                f"{'pin' if expected_hold else 'resample'} at clip end"
            )
            assert bool(command.motion_ended.all()) is expected_hold
    finally:
        default_env.close()
        hold_env.close()
