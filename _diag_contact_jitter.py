"""接触抖动诊断：区分"策略指令在抖" vs "物理仿真在抖"。

用法: uv run --no-sync python _diag_contact_jitter.py <run_name> [steps]
"""
import sys

sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np
import torch
from hydra import compose, initialize

RUN = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=['task=dm10_joystick_flat/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
from unilab.base.registry import ensure_registries
from unilab.base.env_factory import make_registry_env

ensure_registries()
env = make_registry_env('DM10JoystickFlat', 'mujoco', 1, build_ppo_env_cfg_override(cfg))
robot = env.scene['robot']
policy = torch.jit.load(f'logs/rsl_rl_ppo/DM10JoystickFlat/{RUN}/policy.pt', map_location='cpu')
policy.eval()

ctrl_dt = float(env._cfg.ctrl_dt)
phys_dt = float(env._cfg.sim_dt)
decim = int(round(ctrl_dt / phys_dt))
print(f'=== 接触抖动诊断  {RUN} ===')
print(f'  sim_dt={phys_dt}s  ctrl_dt={ctrl_dt}s  decimation={decim}')
print()


def get_obs(o):
    if hasattr(o, 'obs'):
        o = o.obs
    if isinstance(o, dict):
        return np.asarray(o.get('policy', list(o.values())[0]), dtype=np.float32)
    return np.asarray(o, dtype=np.float32)


state = env.reset()
if isinstance(state, tuple):
    state = state[0]

acts, foot_z, foot_v = [], [], []
for t in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    with torch.no_grad():
        a = policy(torch.from_numpy(o)).numpy()
    acts.append(a[0].copy())
    out = env.step(a)
    state = out[0] if isinstance(out, tuple) else out
    d = robot.data
    bodies = list(robot.body_names)
    li, ri = bodies.index('leg_l5_link'), bodies.index('leg_r5_link')
    foot_z.append([d.body_link_pos_w[0, li, 2], d.body_link_pos_w[0, ri, 2]])
    foot_v.append([d.body_link_lin_vel_w[0, li, 2], d.body_link_lin_vel_w[0, ri, 2]])

acts = np.array(acts)      # (T,10)
foot_z = np.array(foot_z)  # (T,2)
foot_v = np.array(foot_v)

print('--- 1. 策略动作的抖动程度 ---')
for i in range(10):
    d1 = np.abs(np.diff(acts[:, i])).mean()
    print(f'  关节{i}: 相邻步动作差均值 {d1:.4f}', end='')
    print(f'  动作范围 [{acts[:,i].min():+.2f},{acts[:,i].max():+.2f}]' if i == 0 else '')
print()
print(f'  全关节平均 |Δaction| = {np.abs(np.diff(acts, axis=0)).mean():.4f}')
print(f'  自相关(1阶) 平均 = {np.mean([np.corrcoef(acts[:-1,i], acts[1:,i])[0,1] for i in range(10)]):+.3f}')

print()
print('--- 2. 脚部高度的抖动 ---')
for i, name in enumerate(['左', '右']):
    z = foot_z[:, i]
    dz = np.abs(np.diff(z))
    sign_flips = int(np.sum(np.diff(np.sign(np.diff(z))) != 0))
    print(f'  {name}脚: z范围 [{z.min():.4f},{z.max():.4f}]  |Δz|均值 {dz.mean():.5f}  方向翻转 {sign_flips} 次 → {sign_flips / (STEPS * ctrl_dt):.1f} Hz')
    vz = foot_v[:, i]
    print(f'       垂直速度范围 [{vz.min():+.3f},{vz.max():+.3f}] m/s  标准差 {vz.std():.3f}')

print()
print('--- 3. 接触判定的稳定性（用 2cm 阈值近似）---')
for i, name in enumerate(['左', '右']):
    z = foot_z[:, i]
    zmin = np.percentile(z, 1)
    contact = z < (zmin + 0.02)
    trans = int(np.sum(contact[:-1] != contact[1:]))
    print(f'  {name}脚: 触地占比 {contact.mean()*100:.1f}%  状态翻转 {trans} 次 → {trans / (STEPS * ctrl_dt):.1f} Hz')
    # 连续接触段长度
    runs = []
    cur = 0
    for c in contact:
        if c:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur: runs.append(cur)
    if runs:
        print(f'       接触段数 {len(runs)}  平均长度 {np.mean(runs)*ctrl_dt*1000:.0f} ms  最长 {max(runs)*ctrl_dt*1000:.0f} ms')

print()
print('--- 4. 判据 ---')
d1_act = np.abs(np.diff(acts, axis=0)).mean()
print(f'  若 |Δaction| > 0.1 且 自相关 < 0.5 → 策略在抖（需 action_rate/dof_acc 惩罚）')
print(f'  若 |Δaction| 小但脚 z 高频翻转 → 物理/接触模型在抖（需查 solver/摩擦/接触刚度）')
print(f'  实测: |Δaction|={d1_act:.4f}')
