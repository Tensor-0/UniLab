"""量化策略的关节摆幅——不依赖视频，直接从仿真读真值。

用法: uv run --no-sync python _measure_hip.py <run_name> [steps]
"""
import sys
sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np, torch
from hydra import compose, initialize

RUN = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=['task=dm10_joystick_flat/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
from unilab.base.registry import ensure_registries
from unilab.base.env_factory import make_registry_env
ensure_registries()

override = build_ppo_env_cfg_override(cfg)
env = make_registry_env('DM10JoystickFlat', 'mujoco', 1, override)

policy = torch.jit.load(f'logs/rsl_rl_ppo/DM10JoystickFlat/{RUN}/policy.pt', map_location='cpu')
policy.eval()

def get_obs(o):
    if hasattr(o, 'obs'): o = o.obs
    if isinstance(o, dict):
        return np.asarray(o.get('policy', list(o.values())[0]), dtype=np.float32)
    return np.asarray(o, dtype=np.float32)

state = env.reset()
if isinstance(state, tuple): state = state[0]

# obs 布局: ang_vel3, gproj3, cmd3, joint_pos10(6:16) 相对默认位, joint_vel10(16:26), act10
names = ['hip_l', 'hip_2', 'hip_3', 'knee_l', 'ankle_l', 'hip_r', 'hip_2r', 'hip_3r', 'knee_r', 'ankle_r']
hist = {n: [] for n in names}
for t in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    with torch.no_grad():
        a = policy(torch.from_numpy(o)).numpy()
    jp = o[0, 6:16]
    for i, n in enumerate(names):
        hist[n].append(float(jp[i]))
    out = env.step(a)
    state = out[0] if isinstance(out, tuple) else out

print(f'=== {RUN} ({STEPS} 步, 单机器人) ===')
print(f'{"关节":10s} {"均值":>8s} {"摆幅p95-p5":>12s} {"范围":>20s}')
for n in names:
    a = np.array(hist[n])
    span = np.percentile(a, 95) - np.percentile(a, 5)
    print(f'{n:10s} {a.mean():+8.3f} {span:12.3f}   [{a.min():+.2f}, {a.max():+.2f}]')

# 关键判据：髋关节是否在摆动
hl = np.array(hist['hip_l']); hr = np.array(hist['hip_r'])
print()
print(f'▶ 左髋摆幅 {np.percentile(hl,95)-np.percentile(hl,5):.3f} rad, 右髋 {np.percentile(hr,95)-np.percentile(hr,5):.3f} rad')
print(f'▶ 左右髋相关性 {np.corrcoef(hl, hr)[0,1]:+.2f}  (接近 -1 = 反相摆动 = 正常交替步态)')
