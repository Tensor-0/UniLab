"""Unit tests for the motion-ended gated stand-still reward.

These exercise the reward through the real MotionCommand type because the module's
``_command`` helper enforces ``isinstance(term, MotionCommand)``, so a stand-in
namespace cannot reach the function under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.tasks.motion_tracking.common import manager_terms as mt


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch):
    """A minimal env whose command term is a MotionCommand *instance*.

    ``_command`` rejects non-MotionCommand terms, so we build the object with
    ``object.__new__`` and only set the attribute the reward reads.
    """
    from types import SimpleNamespace

    num_envs, num_joints = 5, 10
    joint_pos = np.zeros((num_envs, num_joints), dtype=np.float32)
    joint_pos[0, :3] = 0.4
    joint_vel = np.zeros((num_envs, num_joints), dtype=np.float32)
    joint_vel[1, :2] = 1.0
    default = np.zeros((num_envs, num_joints), dtype=np.float32)
    gravity = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (num_envs, 1))
    gravity[3] = np.array([0.0, 0.0, -0.2], dtype=np.float32)  # tilted ~78 deg

    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            default_joint_pos=default,
            projected_gravity_b=gravity,
        )
    )
    ended = np.array([True, True, True, True, False])

    command = object.__new__(mt.MotionCommand)
    # `motion_ended` is a read-only property forwarding to the sampler, so the
    # flag has to be planted on the sampler.
    command.sampler = SimpleNamespace(motion_ended=ended)

    env = SimpleNamespace(
        num_envs=num_envs,
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    return env, joint_pos, joint_vel, gravity


def test_is_zero_when_motion_has_not_ended(scene) -> None:
    env, _, _, _ = scene
    # env 4 is the only non-ended row and it is heavily out of pose.
    values = mt.stand_still_after_motion(env, "motion")
    assert values[4] == 0.0


def test_matches_hand_computed_l1(scene) -> None:
    env, _, _, _ = scene
    values = mt.stand_still_after_motion(env, "motion", pos_weight=1.0, vel_weight=0.04)
    # env 0: |0.4| summed over 3 joints = 1.2, no velocity, upright => 1.2
    assert values[0] == pytest.approx(1.2)
    # env 1: no position error, 2 joints at 1.0 rad/s => 2.0 * 0.04 = 0.08
    assert values[1] == pytest.approx(0.08)
    # env 2: nothing to penalize
    assert values[2] == pytest.approx(0.0)


def test_upright_gate_scales_with_tilt(scene) -> None:
    """The gate is clip(-g_z, 0, 0.7)/0.7.

    Note what this actually does: the gate is 1.0 for any tilt below ~45 deg and
    only starts to close once the body is pushed past that. Past horizontal
    (-g_z <= 0) it is exactly zero, so a robot lying down is not penalized for
    being out of pose.
    """
    env, joint_pos, _, gravity = scene
    # Upright: full weight. error 1.0 * pos_weight 1.0 = 1.0
    assert mt.stand_still_after_motion(env, "motion")[0] == pytest.approx(1.2)
    # Non-upright but still above the clip floor: partially gated.
    gravity[0] = np.array([0.0, 0.0, -0.2], dtype=np.float32)  # -g_z = 0.2
    assert mt.stand_still_after_motion(env, "motion")[0] == pytest.approx(1.2 * 0.2 / 0.7)
    # Lying flat: gate is exactly zero regardless of how far out of pose it is.
    gravity[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert mt.stand_still_after_motion(env, "motion")[0] == pytest.approx(0.0)


def test_does_not_mutate_its_inputs(scene) -> None:
    env, joint_pos, joint_vel, gravity = scene
    before = (joint_pos.copy(), joint_vel.copy(), gravity.copy())
    mt.stand_still_after_motion(env, "motion")
    assert np.array_equal(joint_pos, before[0])
    assert np.array_equal(joint_vel, before[1])
    assert np.array_equal(gravity, before[2])


def test_returns_a_fresh_array(scene) -> None:
    env, joint_pos, _, _ = scene
    values = mt.stand_still_after_motion(env, "motion")
    assert values.shape == (env.num_envs,)
    assert not np.shares_memory(values, joint_pos)


def test_pos_weight_zero_isolates_the_velocity_term(scene) -> None:
    env, _, joint_vel, _ = scene
    values = mt.stand_still_after_motion(env, "motion", pos_weight=0.0, vel_weight=1.0)
    assert values[1] == pytest.approx(float(np.abs(joint_vel[1]).sum()))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pos_weight": -1.0}, "non-negative"),
        ({"vel_weight": -0.1}, "non-negative"),
        ({"upright_gate": 0.0}, "positive"),
        ({"upright_gate": -1.0}, "positive"),
    ],
)
def test_rejects_invalid_gains(scene, kwargs, match) -> None:
    env, _, _, _ = scene
    with pytest.raises(ValueError, match=match):
        mt.stand_still_after_motion(env, "motion", **kwargs)


def test_rejects_non_numeric_gain(scene) -> None:
    env, _, _, _ = scene
    with pytest.raises(TypeError, match="real number"):
        mt.stand_still_after_motion(env, "motion", pos_weight="1.0")  # type: ignore[arg-type]
