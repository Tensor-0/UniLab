"""Render any DM10 policy rollout to MP4 with consistent framing.

Shared renderer for the A/B comparison video: same camera, same resolution,
same background for every policy, so clips can be stacked side by side.

Usage:
    uv run --no-sync python _render_compare.py <task> <run_name> <output.mp4> [steps]
      task: dm10_joystick_flat | dm10_motion_tracking
"""

from __future__ import annotations

import sys

sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np
import torch
import mujoco
from hydra import compose, initialize

TASK = sys.argv[1]
RUN = sys.argv[2]
OUT = sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 600
TASK_NAME = {'dm10_joystick_flat': 'DM10JoystickFlat',
             'dm10_motion_tracking': 'DM10MotionTracking'}[TASK]

# --- shared framing (identical for every clip) ---
W, H, FPS = 640, 360, 50
CAM_DISTANCE, CAM_ELEVATION, CAM_AZIMUTH = 1.6, -4.0, 145.0
CAM_LOOKAT_Z = 0.42   # fixed look-at height (camera does NOT follow the robot)

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=[f'task={TASK}/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
from unilab.base.registry import ensure_registries
from unilab.base.env_factory import make_registry_env

ensure_registries()
env = make_registry_env(TASK_NAME, 'mujoco', 1, build_ppo_env_cfg_override(cfg))
policy = torch.jit.load(f'logs/rsl_rl_ppo/{TASK_NAME}/{RUN}/policy.pt', map_location='cpu')
policy.eval()

backend = env._backend
model = backend._model
qpos_view = backend._qpos_view          # (num_envs, nq)
qvel_view = backend._dof_vel_view       # (num_envs, nv - root)

data = mujoco.MjData(model)
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance, cam.elevation, cam.azimuth = CAM_DISTANCE, CAM_ELEVATION, CAM_AZIMUTH


def get_obs(o):
    if hasattr(o, 'obs'):
        o = o.obs
    if isinstance(o, dict):
        return np.asarray(o.get('policy', list(o.values())[0]), dtype=np.float32)
    return np.asarray(o, dtype=np.float32)


state = env.reset()
if isinstance(state, tuple):
    state = state[0]

frames = []
for t in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    with torch.no_grad():
        a = policy(torch.from_numpy(o)).numpy()
    out = env.step(a)
    state = out[0] if isinstance(out, tuple) else out

    data.qpos[:] = qpos_view[0]
    mujoco.mj_forward(model, data)
    # Follow the robot at a FIXED distance/angle: keeps it centred and the
    # same size in every clip while still showing forward travel.
    cam.lookat[:] = np.array([data.xpos[base_id][0], data.xpos[base_id][1], CAM_LOOKAT_Z])
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())

import imageio.v2 as imageio
imageio.mimsave(OUT, frames, fps=FPS, macro_block_size=None)
print(f"wrote {OUT}  ({len(frames)} frames @ {FPS}fps = {len(frames)/FPS:.1f}s)")
