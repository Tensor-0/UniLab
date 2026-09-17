"""Render a trained policy's rollout to MP4 (single robot, ground-truth driven).

Complements render_motion_mp4.py: that one replays a *reference* motion, this
one runs the *policy* in the sim and films what it actually does.

Usage:
    uv run --no-sync python _render_policy_mp4.py <run_name> [steps] [output.mp4]
"""

from __future__ import annotations

import sys

sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np
import torch
from hydra import compose, initialize

RUN = sys.argv[1] if len(sys.argv) > 1 else None
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
OUT = sys.argv[3] if len(sys.argv) > 3 else '/tmp/policy_rollout.mp4'
TASK, TASK_NAME = 'dm10_motion_tracking', 'DM10MotionTracking'

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=[f'task={TASK}/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
from unilab.base.registry import ensure_registries
from unilab.base.env_factory import make_registry_env

ensure_registries()
env = make_registry_env(TASK_NAME, 'mujoco', 1, build_ppo_env_cfg_override(cfg))
robot = env.scene['robot']
policy = torch.jit.load(f'logs/rsl_rl_ppo/{TASK_NAME}/{RUN}/policy.pt', map_location='cpu')
policy.eval()


def get_obs(o):
    if hasattr(o, 'obs'):
        o = o.obs
    if isinstance(o, dict):
        return np.asarray(o.get('policy', list(o.values())[0]), dtype=np.float32)
    return np.asarray(o, dtype=np.float32)


state = env.reset()
if isinstance(state, tuple):
    state = state[0]

import mujoco
model = env._backend._model
data = env._backend._data
renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance, cam.elevation, cam.azimuth = 2.4, -8.0, 135.0
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

frames = []
for t in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    with torch.no_grad():
        a = policy(torch.from_numpy(o)).numpy()
    out = env.step(a)
    state = out[0] if isinstance(out, tuple) else out
    mujoco.mj_forward(model, data)
    cam.lookat[:] = data.xpos[base_id]
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())

import imageio.v2 as imageio
imageio.mimsave(OUT, frames, fps=50, macro_block_size=None)
print(f"wrote {OUT}  ({len(frames)} frames)")
