"""步态量化：脚底高度 / 腾空时间 / 接触滑移，只读仿真真值。

用法: uv run --no-sync python _measure_gait.py <run_name> [steps]
"""
import sys

sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np
import torch
from hydra import compose, initialize

RUN = sys.argv[1] if len(sys.argv) > 1 else '2026-09-09_03-57-07_mujoco'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
AIR_THRESHOLD = 0.02  # m，高于最低点 2cm 记为腾空

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=['task=dm10_joystick_flat/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
from unilab.base.registry import ensure_registries
from unilab.base.env_factory import make_registry_env

ensure_registries()
env = make_registry_env('DM10JoystickFlat', 'mujoco', 1, build_ppo_env_cfg_override(cfg))
policy = torch.jit.load(f'logs/rsl_rl_ppo/DM10JoystickFlat/{RUN}/policy.pt', map_location='cpu')
policy.eval()
robot = env.scene['robot']
step_dt = float(getattr(env, 'step_dt', 0.01))
cmd_term = env.command_manager.get_term('twist')

bodies = list(robot.body_names)
print(f'=== 步态量化  {RUN}  ({STEPS} 步 = {STEPS * step_dt:.1f}s, dt={step_dt}) ===')
print(f'注册的 body: {bodies}')
li, ri = bodies.index('leg_l5_link'), bodies.index('leg_r5_link')


def get_obs(o):
    if hasattr(o, 'obs'):
        o = o.obs
    if isinstance(o, dict):
        return np.asarray(o.get('policy', list(o.values())[0]), dtype=np.float32)
    return np.asarray(o, dtype=np.float32)


state = env.reset()
if isinstance(state, tuple):
    state = state[0]

feet, base = [], []
for _ in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    with torch.no_grad():
        a = policy(torch.from_numpy(o)).numpy()
    out = env.step(a)
    state = out[0] if isinstance(out, tuple) else out
    feet.append(robot.data.body_link_pos_w[0].copy())
    base.append(robot.data.root_link_pos_w[0].copy())

feet = np.array(feet)          # (T, n_bodies, 3)
base = np.array(base)
print(f'base 前进速度(体坐标真值): {np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1).mean() / step_dt:.3f} m/s '
      f'(净位移 {np.linalg.norm(base[-1, :2] - base[0, :2]):.2f} m)')
print()

for label, idx in (('左腿 leg_l5_link', li), ('右腿 leg_r5_link', ri)):
    f = feet[:, idx, :]
    z = f[:, 2]
    zmin = float(np.percentile(z, 1))
    clearance = float(z.max() - zmin)
    air = z > (zmin + AIR_THRESHOLD)
    air_frac = float(air.mean())
    # 腾空段计数 → 步频
    trans = int(np.sum((~air[:-1]) & air[1:]))
    freq = trans / (STEPS * step_dt) if trans else float('nan')
    # 接触期水平滑移速度（蹭地指标）
    dxy = np.linalg.norm(np.diff(f[:, :2], axis=0), axis=1) / step_dt
    contact = ~air[:-1]
    slip = float(dxy[contact].mean()) if contact.sum() > 2 else float('nan')
    print(f'--- {label} ---')
    print(f'  最低点 z={zmin:.4f}  最高点 z={z.max():.4f}  抬脚高度(最大-最低) {clearance * 100:.1f} cm')
    print(f'  腾空占比 {air_frac * 100:.1f}%   (阈值: 高于最低点 {AIR_THRESHOLD * 100:.0f} cm)')
    print(f'  腾空段数 {trans}  →  步频 {freq:.2f} Hz  (周期 {1 / freq:.2f} s)' if trans else '  全程无腾空')
    print(f'  接触期水平滑移速度 {slip:.3f} m/s   (蹭地指标：越小越好)')
    print(f'  脚 z 范围 [{z.min():.3f}, {z.max():.3f}]')
    print()

print('--- 对照：base 高度与速度 ---')
print(f'base z 均值 {base[:, 2].mean():.3f} 标准差 {base[:, 2].std():.4f}')
print(f'指令 vx {cmd_term.command[0, 0]:.2f}')
