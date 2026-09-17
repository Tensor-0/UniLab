"""feet_contact_number 相位诊断：验证相位推进、reset 行为、以及与真实步态的匹配度。

用法: uv run --no-sync python _diag_contact_phase.py [run_name] [steps]
     不传 run_name 时用随机策略（检验相位机制本身）
"""
import sys

sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np
import torch
from hydra import compose, initialize

RUN = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else None
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
CYCLE = 0.64

with initialize(version_base=None, config_path='src/unilab/conf/ppo'):
    cfg = compose(config_name='config', overrides=['task=dm10_joystick_flat/mujoco'])

from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
from unilab.base.registry import ensure_registries
from unilab.base.env_factory import make_registry_env

ensure_registries()
env = make_registry_env('DM10JoystickFlat', 'mujoco', 1, build_ppo_env_cfg_override(cfg))
robot = env.scene['robot']
step_dt = float(getattr(env, 'step_dt', 0.02))

policy = None
if RUN:
    policy = torch.jit.load(f'logs/rsl_rl_ppo/DM10JoystickFlat/{RUN}/policy.pt', map_location='cpu')
    policy.eval()

term = env.reward_manager.get_term_cfg('feet_contact_number')
fn = term.func
print(f'=== 相位诊断 {"run=" + RUN if RUN else "随机策略"} ({STEPS} 步 @ {step_dt}s) ===')
print(f'cycle_time = {fn._cycle_time}s  →  期望步频 {1.0 / fn._cycle_time:.2f} Hz')
print(f'double_support_band = {fn._band}, mismatch_score = {fn._mismatch_score}')
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

# 记录：相位时间、sin、目标触地、实际触地
rec = []
for t in range(STEPS):
    o = get_obs(state).reshape(1, -1)
    if policy is not None:
        with torch.no_grad():
            a = policy(torch.from_numpy(o)).numpy()
    else:
        a = np.random.randn(1, 10).astype(np.float32) * 0.5
    out = env.step(a)
    state = out[0] if isinstance(out, tuple) else out
    contact = fn._contact(env)[0].copy()
    sin_pos = np.sin(2 * np.pi * fn._phase_time[0] / CYCLE)
    rec.append((fn._phase_time[0], sin_pos, contact))

pt = np.array([r[0] for r in rec])
sinp = np.array([r[1] for r in rec])
ct = np.array([r[2] for r in rec])       # (T,2) [left, right]

print(f'--- 1. 相位推进 ---')
print(f'  相位时间范围 [{pt.min():.2f}, {pt.max():.2f}]s   单调递增: {bool(np.all(np.diff(pt) >= -1e-9))}')
print(f'  sin 范围 [{sinp.min():+.3f}, {sinp.max():+.3f}]  （应覆盖 -1~+1）')
crossings = int(np.sum(np.diff(np.sign(sinp)) != 0))
print(f'  过零点次数 {crossings}  →  实测周期 {2 * STEPS * step_dt / max(crossings, 1):.2f}s')

print()
print(f'--- 2. 实际触地统计 ---')
for i, name in enumerate(['左腿', '右腿']):
    frac = ct[:, i].mean()
    trans = int(np.sum((~ct[:-1, i]) & ct[1:, i]))
    print(f'  {name}: 触地占比 {frac * 100:.1f}%  触地次数 {trans}  → 步频 {trans / (STEPS * step_dt):.2f} Hz')
both = np.logical_and(ct[:, 0], ct[:, 1]).mean()
neither = np.logical_and(~ct[:, 0], ~ct[:, 1]).mean()
print(f'  双脚同时触地 {both * 100:.1f}%   双脚同时离地 {neither * 100:.1f}%')

print()
print(f'--- 3. 接触-相位匹配率（这是奖励关心的）---')
target = np.stack([sinp >= 0, sinp < 0], axis=1)
target[np.abs(sinp) < fn._band] = True
match = (ct == target).mean(axis=1)
print(f'  平均匹配率 {match.mean() * 100:.1f}%  （50% = 随机水平，>70% 说明策略在迎合相位）')
for i, name in enumerate(['左腿', '右腿']):
    print(f'  {name}匹配率 {(ct[:, i] == target[:, i]).mean() * 100:.1f}%')

print()
print(f'--- 4. 与真实步频对比 ---')
freqs = []
for i in range(2):
    sig = ct[:, i].astype(float) - ct[:, i].mean()
    f = np.fft.rfftfreq(len(sig), d=step_dt)
    sp = np.abs(np.fft.rfft(sig))
    if len(sp) > 2 and sp[1:].max() > 0:
        k = int(np.argmax(sp[1:]) + 1)
        freqs.append(f[k])
        print(f'  {"左腿" if i == 0 else "右腿"}触地主频 {f[k]:.2f} Hz  (相位要求 {1 / CYCLE:.2f} Hz)')
if len(freqs) == 2:
    print(f'  左/右步频比 {freqs[0] / max(freqs[1], 1e-6):.2f}  （1.0 = 对称）')
