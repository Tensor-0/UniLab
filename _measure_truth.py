"""判决性测量：只读仿真真值，量化策略的实际位移/速度/关节摆幅。

与 _measure_hip.py / _measure_joints.py 的区别：关节角与速度一律从 `robot.data`
（物理引擎真值）读取，同时把 obs 向量里的对应切片并排打印，用来检验「obs 布局错位」。

用法: uv run --no-sync python _measure_truth.py <run_name> [steps]
"""
import sys

sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np
import torch
from hydra import compose, initialize

RUN = sys.argv[1] if len(sys.argv) > 1 else '2026-09-09_03-57-07_mujoco'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TASK = sys.argv[3] if len(sys.argv) > 3 else 'dm10_joystick_flat'
TASK_NAME = {'dm10_joystick_flat': 'DM10JoystickFlat', 'dm10_motion_tracking': 'DM10MotionTracking'}.get(TASK, 'DM10JoystickFlat')

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=[f'task={TASK}/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
from unilab.base.registry import ensure_registries
from unilab.base.env_factory import make_registry_env

ensure_registries()
env = make_registry_env(TASK_NAME, 'mujoco', 1, build_ppo_env_cfg_override(cfg))
policy = torch.jit.load(f'logs/rsl_rl_ppo/{TASK_NAME}/{RUN}/policy.pt', map_location='cpu')
policy.eval()
robot = env.scene['robot']
cmd_term = env.command_manager.get_term('twist')
step_dt = float(getattr(env, 'step_dt', 0.01))


def get_obs(o):
    if hasattr(o, 'obs'):
        o = o.obs
    if isinstance(o, dict):
        return np.asarray(o.get('policy', list(o.values())[0]), dtype=np.float32)
    return np.asarray(o, dtype=np.float32)


state = env.reset()
if isinstance(state, tuple):
    state = state[0]
o0 = get_obs(state).reshape(1, -1)
names = list(robot.joint_names) if hasattr(robot, 'joint_names') else [f'j{i}' for i in range(10)]

print(f'=== 判决性测量  {RUN}  ({STEPS} 步 @ {step_dt}s = {STEPS * step_dt:.1f}s) ===')
print(f'obs 维度   = {o0.shape[1]}')
print(f'关节真值顺序 = {names}')
print(f'初始指令   = {np.round(cmd_term.command[0], 3)}')

rec = []
for t in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    with torch.no_grad():
        a = policy(torch.from_numpy(o)).numpy()
    out = env.step(a)
    state = out[0] if isinstance(out, tuple) else out
    d = robot.data
    rec.append((d.root_link_pos_w[0].copy(), d.root_link_lin_vel_b[0].copy(),
                d.joint_pos[0].copy(), o[0, 6:16].copy(), cmd_term.command[0].copy()))

pos = np.array([r[0] for r in rec])
vel = np.array([r[1] for r in rec])
jt = np.array([r[2] for r in rec])
ojp = np.array([r[3] for r in rec])
cmd = np.array([r[4] for r in rec])

print()
print('--- 位移（真值）---')
print(f'起点 xy ({pos[0, 0]:+.3f}, {pos[0, 1]:+.3f})  →  终点 xy ({pos[-1, 0]:+.3f}, {pos[-1, 1]:+.3f})')
print(f'净位移 {np.linalg.norm(pos[-1, :2] - pos[0, :2]):.3f} m    '
      f'路径长 {np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)):.3f} m')
print(f'高度 z 均值 {pos[:, 2].mean():.3f}  标准差 {pos[:, 2].std():.4f}')

print()
print('--- 速度跟踪（真值）---')
print(f'指令 vx   均值 {cmd[:, 0].mean():+.3f}  范围 [{cmd[:, 0].min():+.2f}, {cmd[:, 0].max():+.2f}]')
print(f'实际 vx_b 均值 {vel[:, 0].mean():+.3f}  范围 [{vel[:, 0].min():+.2f}, {vel[:, 0].max():+.2f}]')
err = np.linalg.norm(cmd[:, :2] - vel[:, :2], axis=1)
print(f'平均 |速度误差| = {err.mean():.3f} m/s   中位数 {np.median(err):.3f}')

print()
print('--- 关节摆幅（真值 joint_pos）---')
for i, n in enumerate(names):
    a = jt[:, i]
    print(f'{n:22s} 均值 {a.mean():+7.3f}  摆幅(p95-p5) {np.percentile(a, 95) - np.percentile(a, 5):6.3f}  '
          f'范围 [{a.min():+.2f}, {a.max():+.2f}]')

print()
print('--- obs[6:16] 切片 vs 真值（检验观测对齐）---')
for i, n in enumerate(names):
    a, b = jt[:, i], ojp[:, i]
    c = np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 and b.std() > 1e-9 else float('nan')
    print(f'{n:22s} 真值摆幅 {np.percentile(a, 95) - np.percentile(a, 5):6.3f}   '
          f'obs摆幅 {np.percentile(b, 95) - np.percentile(b, 5):6.3f}   相关 {c:+.2f}')

print()
print('--- 步频估计（髋关节真值角 FFT）---')
for i in (0, min(5, len(names) - 1)):
    hip = jt[:, i] - jt[:, i].mean()
    f = np.fft.rfftfreq(len(hip), d=step_dt)
    sp = np.abs(np.fft.rfft(hip))
    if len(sp) > 2:
        k = int(np.argmax(sp[1:]) + 1)
        print(f'{names[i]:22s} 主频 {f[k]:.2f} Hz  周期 {1 / f[k]:.2f} s')
