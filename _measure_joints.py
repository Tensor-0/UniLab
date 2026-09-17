"""量化策略的关节摆幅——不依赖视频，直接从仿真读真值。

用法:
    uv run --no-sync python _measure_joints.py <task> <run_name> <joint_csv> [steps]

例:
    uv run --no-sync python _measure_joints.py dm10_joystick_flat 2026-09-09_03-10-01_mujoco \
        leg_l1_joint,leg_r1_joint,leg_l4_joint,leg_r4_joint 400

输出关节的均值/摆幅/范围，以及左右配对的相关性（接近 -1 = 反相摆动 = 正常交替步态）。
"""
import sys
sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np
import torch
from hydra import compose, initialize

TASK = sys.argv[1]
RUN = sys.argv[2]
JOINTS = [j for j in sys.argv[3].split(',') if j]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 400

TASK_NAMES = {
    'dm10_joystick_flat': 'DM10JoystickFlat',
    'g1_walk_flat': 'G1WalkFlat',
    'go2_joystick_flat': 'Go2JoystickFlat',
}

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=[f'task={TASK}/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
from unilab.base.registry import ensure_registries
from unilab.base.env_factory import make_registry_env

ensure_registries()
task_name = TASK_NAMES.get(TASK, cfg.training.task_name)
env = make_registry_env(task_name, 'mujoco', 1, build_ppo_env_cfg_override(cfg))

policy = torch.jit.load(f'logs/rsl_rl_ppo/{task_name}/{RUN}/policy.pt', map_location='cpu')
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

# 从场景里解析关节在 joint_pos 观测块中的下标。
# 观测布局 dm10: ang_vel3, gproj3, cmd3, joint_pos10, joint_vel10, act10
entity = env.scene['robot']
all_joints = list(entity.data.joint_names) if hasattr(entity.data, 'joint_names') else None
if all_joints is None:
    raise SystemExit('无法读取关节名列表')

OBS_JP_START = 6  # ang_vel(3) + projected_gravity(3)
idx = []
for j in JOINTS:
    if j not in all_joints:
        raise SystemExit(f'关节 {j} 不在 {all_joints}')
    idx.append(all_joints.index(j))

hist = {j: [] for j in JOINTS}
for _ in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    with torch.no_grad():
        action = policy(torch.from_numpy(o)).numpy()
    jp = o[0, OBS_JP_START:OBS_JP_START + len(all_joints)]
    for j, i in zip(JOINTS, idx):
        hist[j].append(float(jp[i]))
    out = env.step(action)
    state = out[0] if isinstance(out, tuple) else out

print(f'=== {task_name} / {RUN} ({STEPS} 步, 单机器人) ===')
print(f'{"关节":28s} {"均值":>8s} {"摆幅p95-p5":>12s} {"范围":>22s}')
stats = {}
for j in JOINTS:
    a = np.array(hist[j])
    span = np.percentile(a, 95) - np.percentile(a, 5)
    stats[j] = a
    print(f'{j:28s} {a.mean():+8.3f} {span:12.3f}   [{a.min():+.2f}, {a.max():+.2f}]')

# 左右配对相关性
if len(JOINTS) >= 2 and len(JOINTS) % 2 == 0:
    half = len(JOINTS) // 2
    for jl, jr in zip(JOINTS[:half], JOINTS[half:]):
        c = np.corrcoef(stats[jl], stats[jr])[0, 1]
        flag = '✅ 反相' if c < -0.5 else ('⚠️ 同相' if c > 0.5 else '— 弱相关')
        print(f'▶ {jl} vs {jr}: 相关性 {c:+.2f}  {flag}')
