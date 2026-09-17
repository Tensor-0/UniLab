"""Measure a motion-tracking policy using ground truth only.

Unlike _measure_truth.py (written for the joystick task), this reads the
`motion` command instead of `twist` and reports how well the policy tracks the
reference trajectory, plus the gait-quality metrics that matter for DM10.

Usage: uv run --no-sync python _measure_mt.py <run_name> [steps]
"""
import sys

sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np
import torch
from hydra import compose, initialize

RUN = sys.argv[1] if len(sys.argv) > 1 else None
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
TASK, TASK_NAME = 'dm10_motion_tracking', 'DM10MotionTracking'

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=[f'task={TASK}/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
from unilab.base.registry import ensure_registries
from unilab.base.env_factory import make_registry_env

ensure_registries()
env = make_registry_env(TASK_NAME, 'mujoco', 1, build_ppo_env_cfg_override(cfg))
robot = env.scene['robot']
step_dt = float(getattr(env, 'step_dt', 0.02))
cmd = env.command_manager.get_term('motion')

policy = None
if RUN:
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

print(f'=== motion tracking 测量 {RUN or "随机策略"} ({STEPS} 步 @ {step_dt}s) ===')
bodies = list(robot.body_names)
li, ri = bodies.index('leg_l5_link'), bodies.index('leg_r5_link')
base = bodies.index('base_link')

pos, ref_pos, jt, ref_jt, cmd_v = [], [], [], [], []
for t in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    if policy is not None:
        with torch.no_grad():
            a = policy(torch.from_numpy(o)).numpy()
    else:
        a = np.random.randn(1, 10).astype(np.float32) * 0.3
    out = env.step(a)
    state = out[0] if isinstance(out, tuple) else out
    d = robot.data
    pos.append(d.body_link_pos_w[0, [base, li, ri]].copy())
    jt.append(d.joint_pos[0].copy())
    # reference (motion command) state
    ref_pos.append(cmd.body_pos_w[0, [base, li, ri]].copy())
    ref_jt.append(cmd.joint_pos[0].copy())

pos = np.array(pos); ref_pos = np.array(ref_pos)
jt = np.array(jt); ref_jt = np.array(ref_jt)

print()
print('--- 1. 步态质量（真值）---')
for i, nm in enumerate(['左腿', '右腿']):
    z = pos[:, i + 1, 2]
    zmin = float(np.percentile(z, 1))
    air = z > zmin + 0.02
    dxy = np.linalg.norm(np.diff(pos[:, i + 1, :2], axis=0), axis=1) / step_dt
    trans = int(np.sum((~air[:-1]) & air[1:]))
    print(f'  {nm}: 抬脚 {100 * (z.max() - zmin):.1f} cm  腾空 {100 * air.mean():.1f}%  '
          f'步频 {trans / (STEPS * step_dt):.2f} Hz  接触滑移 {dxy[~air[:-1]].mean() if (~air[:-1]).sum() > 2 else float("nan"):.3f} m/s')

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
    print(f'  策略 {nm} 主频 {f[k]:.2f} Hz', end='')
    rs = ref_jt[:, 3 + 5 * i] - ref_jt[:, 3 + 5 * i].mean()
    rsp = np.abs(np.fft.rfft(rs))
    rk = int(np.argmax(rsp[1:]) + 1)
    print(f'   参考 {nm} 主频 {f[rk]:.2f} Hz')
