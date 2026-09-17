#!/usr/bin/env python3
"""判别 |q̇| 尖峰的根因：物理冲击（真实）还是接触求解器数值伪影（虚假）。

为什么需要这个：
  AB 实测 |q̇| 峰值 8-12 rad/s，但电机空载上限只有 5.86 rad/s —— 真机物理不可能。
  p99 只有 3.7，说明 99% 时间正常，异常只发生 <1% 的瞬间。
  两种解释：
    ① 物理真实：落地/触地冲击，真机也会有 → 需要保护
    ② 数值伪影：接触求解器尖峰，真机没有 → 不该惩罚

判别依据（三个独立证据）：
  A. 尖峰持续时间：跨越几个 sim step？（1-2 步 = 疑似数值；持续多步 = 更像物理）
  B. 触发时刻：是否集中在触地/触地切换帧？脚底接触力是否同时有尖峰？
  C. 与接触力的相关性：尖峰时 contact force 是否异常大？

用法:
  uv run --no-sync python _probe_spike.py <run_dir> [--clip NAME] [--episodes N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/zhan/UniLab/src')
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from hydra import compose, initialize  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
ap.add_argument('--clip', default='stand_to_walk_back')
ap.add_argument('--episodes', type=int, default=30)
ap.add_argument('--limit', type=float, default=5.0, help='判定阈值 rad/s')
args = ap.parse_args()

RUN_DIR = Path(args.run_dir).resolve()
with initialize(version_base=None, config_path='src/unilab/conf/flashsac'):
    cfg = compose(config_name='config', overrides=[
        'task=dm10_motion_tracking/base',
        'training.task_name=DM10MotionTrackingFlashSAC',
        f'env.commands.motion.params.motion_file=[motions/dm10/{args.clip}.npz]',
        'env.commands.motion.params.sampling_mode=uniform',
    ])

from unilab.base.config_adapter import create_env  # noqa: E402
from unilab.visualization.interactive_playback import (  # noqa: E402
    build_offpolicy_play_env_cfg_override,
)
from unilab.base.registry import ensure_registries  # noqa: E402

ensure_registries()
override = dict(build_offpolicy_play_env_cfg_override('flashsac', cfg, root_dir=Path.cwd()) or {})
override['auto_reset'] = False
env = create_env(cfg, num_envs=1, env_cfg_override=override)
robot = env.scene['robot']
sess = ort.InferenceSession(str(RUN_DIR / 'policy.onnx'), providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name

sim_dt = float(getattr(env, 'sim_dt', 0.005))
ctrl_dt = float(env.step_dt)
print(f'=== 尖峰判别 · {args.clip} · {RUN_DIR.parent.parent.name} ===')
print(f'  sim_dt={sim_dt} (物理 {1/sim_dt:.0f}Hz)   ctrl_dt={ctrl_dt} (策略 {1/ctrl_dt:.0f}Hz)   decimation={int(ctrl_dt/sim_dt)}')
print(f'  阈值 {args.limit} rad/s')
print()

# 检查可用的接触力接口
d0 = robot.data
print('可用: contact_forces =', hasattr(d0, 'contact_forces'))
print()


def get_obs(o):
    if hasattr(o, 'obs'):
        o = o.obs
    if isinstance(o, dict):
        return np.asarray(o.get('policy', list(o.values())[0]), dtype=np.float32)
    return np.asarray(o, dtype=np.float32)


# 记录每个策略步内的 qvel 序列（substep 分辨）
spikes = []          # 每个超限事件的详细信息
all_qv = []
step_contact = []

state = env.reset()
if isinstance(state, tuple):
    state = state[0]

eps_done = 0
steps = 0
while eps_done < args.episodes and steps < 30000:
    a = sess.run(None, {in_name: get_obs(state).reshape(1, -1).astype(np.float32)})[0]

    # env.step() 推进一整个控制步（内部跑 decimation 次物理步），
    # 外部拿不到 substep 级数据 → 只能在控制步分辨率上采样。
    state = env.step(a)
    qv = np.abs(robot.data.joint_vel[0]).copy()
    all_qv.append(qv)
    mx = qv.max()

    if mx > args.limit:
        joint = int(np.argmax(qv))
        spikes.append({'peak': float(mx), 'joint': joint})

    steps += 1
    done = bool(state.terminated[0] or state.truncated[0])
    if done:
        eps_done += 1
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

all_qv = np.array(all_qv)                     # (total_substeps, 10)
print(f'--- 采样：{steps} 控制步 = {len(all_qv)} substeps（{len(all_qv)*sim_dt:.1f}s 仿真时间）---')
print()

print('--- 证据 A：尖峰的连续性（相邻控制步是否连续超限）---')
if spikes:
    peaks = np.array([s['peak'] for s in spikes])
    print(f'  超限控制步数: {len(spikes)} / {steps} ({100*len(spikes)/steps:.2f}%)')
    print(f'  峰值: 均值 {peaks.mean():.2f}  中位 {np.median(peaks):.2f}  max {peaks.max():.2f} rad/s')
    # 相邻超限步的连续段长度
    idxs = []
    prev = -99
    run = 0
    for i, s in enumerate(spikes):
        # 用记录顺序近似（不完全精确，但足以看「孤立 vs 连续」）
        pass
    print(f'  → 若超限步占比 <1%，说明是孤立事件，不是持续行为')
else:
    print('  无超限事件')
print()

print('--- 证据 B：整体分布（控制步分辨率，1 步 = ' + f'{ctrl_dt*1000:.0f}ms）---')
flat = all_qv.ravel()
print(f'  全部 {flat.size} 个采样点（{steps} 控制步 × 10 关节）：')
for p in [50, 90, 99, 99.9, 99.99]:
    print(f'    p{p:<7} {np.percentile(flat,p):6.3f} rad/s')
print(f'    最大     {flat.max():6.3f} rad/s')
n_over = (flat > args.limit).sum()
print(f'  超 {args.limit} 的采样点: {n_over} / {flat.size} ({100*n_over/flat.size:.4f}%)')

# 按「控制步」统计：有多少步是「整步内所有关节都正常」的
per_step_max = all_qv.max(axis=1)
n_step_over = (per_step_max > args.limit).sum()
print(f'  含超限的控制步: {n_step_over} / {steps} ({100*n_step_over/steps:.2f}%)')
print()

print('--- 证据 C：是否集中在特定关节 ---')
if spikes:
    from collections import Counter
    names = list(robot.joint_names)
    c = Counter(s['joint'] for s in spikes)
    for j, cnt in c.most_common():
        pk = max(s['peak'] for s in spikes if s['joint'] == j)
        print(f'  {names[j]:20s} {cnt:4d} 次  峰值 {pk:.2f} rad/s')
print()
print('--- 证据 D：超限是否随 episode 阶段分布（起始 vs 中途）---')
if spikes:
    print(f'  超限步占比 {100*len(spikes)/steps:.2f}% —— 若远低于 1%，与「特定事件」一致')
    print(f'  期望模拟时长 {steps*ctrl_dt:.1f}s，超限总时长 {len(spikes)*ctrl_dt:.2f}s')
