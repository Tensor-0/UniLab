#!/usr/bin/env python3
"""用真值测量 FlashSAC / 任意 ONNX 策略的运动跟踪质量（AB 对比用）。

为什么需要这个：现有的 `_measure_mt.py` 只加载 rsl_rl 的 `policy.pt`（JIT），
而 AB 实验是 FlashSAC，产物是 `policy.onnx`。本脚本复用 `_measure_mt.py` 的
全部测量逻辑，只把策略加载换成 ONNX。

用法:
  uv run --no-sync python _measure_mt_onnx.py <run_dir> [steps]
  uv run --no-sync python _measure_mt_onnx.py \
      /home/zhan/UniLab/logs/AB_A_long/DM10MotionTrackingFlashSAC/2026-09-12_15-01-19_mujoco 800
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from hydra import compose, initialize  # noqa: E402

RUN_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
TASK, TASK_NAME = 'dm10_motion_tracking', 'DM10MotionTracking'

if RUN_DIR is None or not (RUN_DIR / 'policy.onnx').exists():
    print(f'❌ 需要 run 目录（含 policy.onnx）: {RUN_DIR}')
    raise SystemExit(1)

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=[f'task={TASK}/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override  # noqa: E402
from unilab.base.registry import ensure_registries  # noqa: E402
from unilab.base.env_factory import make_registry_env  # noqa: E402

ensure_registries()
env = make_registry_env(TASK_NAME, 'mujoco', 1, build_ppo_env_cfg_override(cfg))
robot = env.scene['robot']
step_dt = float(getattr(env, 'step_dt', 0.02))
cmd = env.command_manager.get_term('motion')

session = ort.InferenceSession(str(RUN_DIR / 'policy.onnx'), providers=['CPUExecutionProvider'])
in_name = session.get_inputs()[0].name


def policy(obs: np.ndarray) -> np.ndarray:
    return session.run(None, {in_name: obs.astype(np.float32)})[0]


def get_obs(o):
    if hasattr(o, 'obs'):
        o = o.obs
    if isinstance(o, dict):
        return np.asarray(o.get('policy', list(o.values())[0]), dtype=np.float32)
    return np.asarray(o, dtype=np.float32)


state = env.reset()
if isinstance(state, tuple):
    state = state[0]

print(f'=== 运动跟踪测量 · {RUN_DIR.name} ({STEPS} 步 @ {step_dt}s) ===')
bodies = list(robot.body_names)
li, ri = bodies.index('leg_l5_link'), bodies.index('leg_r5_link')
base = bodies.index('base_link')

pos, ref_pos, jt, ref_jt = [], [], [], []
qvel, tau = [], []
for t in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    a = policy(o)
    out = env.step(a)
    state = out[0] if isinstance(out, tuple) else out
    d = robot.data
    pos.append(d.body_link_pos_w[0, [base, li, ri]].copy())
    jt.append(d.joint_pos[0].copy())
    ref_pos.append(cmd.body_pos_w[0, [base, li, ri]].copy())
    ref_jt.append(cmd.joint_pos[0].copy())
    qvel.append(d.joint_vel[0].copy())

pos = np.array(pos); ref_pos = np.array(ref_pos)
jt = np.array(jt); ref_jt = np.array(ref_jt)
qvel = np.array(qvel)

print()
print('--- 1. 步态质量（真值）---')
for i, nm in enumerate(['左腿', '右腿']):
    z = pos[:, i + 1, 2]
    zmin = float(np.percentile(z, 1))
    air = z > zmin + 0.02
    dxy = np.linalg.norm(np.diff(pos[:, i + 1, :2], axis=0), axis=1) / step_dt
    trans = int(np.sum((~air[:-1]) & air[1:]))
    slip = dxy[~air[:-1]].mean() if (~air[:-1]).sum() > 2 else float('nan')
    print(f'  {nm}: 抬脚 {100 * (z.max() - zmin):.1f} cm  腾空 {100 * air.mean():.1f}%  '
          f'步频 {trans / (STEPS * step_dt):.2f} Hz  接触滑移 {slip:.3f} m/s')

print()
print('--- 2. 位移 ---')
net = np.linalg.norm(pos[-1, 0, :2] - pos[0, 0, :2])
path = np.sum(np.linalg.norm(np.diff(pos[:, 0, :2], axis=0), axis=1))
print(f'  净位移 {net:.2f} m   路径长 {path:.2f} m   高度 z {pos[:, 0, 2].mean():.3f} ± {pos[:, 0, 2].std():.4f}')
print(f'  平均速度 {net / (STEPS * step_dt):.2f} m/s')

print()
print('--- 3. 参考跟踪误差（相对锚点）---')
rel_err = np.linalg.norm((pos - pos[:, :1]) - (ref_pos - ref_pos[:, :1]), axis=2)
for i, nm in enumerate(['base', '左脚', '右脚']):
    print(f'  {nm:6s} 平均位置误差 {rel_err[:, i].mean():.3f} m')
jerr = np.abs(jt - ref_jt).mean()
print(f'  关节角平均误差 {jerr:.3f} rad')

print()
print('--- 4. 步频对比 ---')
for i, nm in enumerate(['左膝', '右膝']):
    sig = jt[:, 3 + 5 * i] - jt[:, 3 + 5 * i].mean()
    f = np.fft.rfftfreq(len(sig), d=step_dt)
    sp = np.abs(np.fft.rfft(sig))
    k = int(np.argmax(sp[1:]) + 1)
    rs = ref_jt[:, 3 + 5 * i] - ref_jt[:, 3 + 5 * i].mean()
    rsp = np.abs(np.fft.rfft(rs))
    rk = int(np.argmax(rsp[1:]) + 1)
    print(f'  策略 {nm} 主频 {f[k]:.2f} Hz   参考 {nm} 主频 {f[rk]:.2f} Hz')

print()
print('--- 5. 硬件负载（本次新增，供 AB 对比）---')
print(f'  关节速度峰值 |q̇|max  {np.abs(qvel).max():.2f} rad/s')
print(f'  关节速度 95 分位      {np.percentile(np.abs(qvel), 95):.2f} rad/s')
print(f'  关节速度均值          {np.abs(qvel).mean():.2f} rad/s')
