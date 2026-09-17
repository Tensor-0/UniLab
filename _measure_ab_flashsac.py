#!/usr/bin/env python3
"""AB 对比测量：FlashSAC 运动跟踪策略的 per-episode 真值指标。

本脚本修过三轮错误，每轮都记录在此（避免重犯）：

第一轮（`_measure_mt_onnx.py`）：用了 PPO 的配置（`conf/ppo/`）去测 FlashSAC 策略。
  两套配置是不同的 contract：flashsac 有 `motion_clip_end` 终止 +
  `truncate_on_clip_end: true` + `sim_dt: 0.005`；PPO 没有，且 false。
  后果：片段结束会 `_resample_command()` —— 参考运动**瞬移到另一个随机片段**，
  位移/速度指标全部失真（测出「净位移 0.13m」的伪影）。

第二轮：`sampling_mode: adaptive` 时片段从任意帧开始，episode 长度与「抽到哪一帧」
  混淆 —— 不控制这个，A/B 就不可比。→ 改 `sampling_mode=uniform` + 固定片段。

第三轮（本版修复，2026-09-12 晚）：
  ① **seed 实际没固定** —— `env_cfg_override` 是**替换**而非合并，之前没把 seed 放进去，
     env 回退到 `secrets.randbits(63)`，结果不可复现。现在显式传 `seed`。
  ② **DR 被意外打开** —— 用训练配置 compose，`env.events` 全部生效，测量数字被
     质量/摩擦/推送/零位偏置污染。现在显式 `events={}` 关掉，测**干净**的策略能力。
  ③ **单 env 评测失真** —— 单 env 下 episode 长度只有 1.0s，而 32 env 下是 3.78s
     （训练记录 3.86s）。差 3.8 倍！现在默认 `num_envs=32` 对齐训练。

用法:
  uv run --no-sync python _measure_ab_flashsac.py <run_dir> [--clip NAME] [--episodes N]
  例:
  uv run --no-sync python _measure_ab_flashsac.py \
      /home/zhan/UniLab/logs/AB_A_long/DM10MotionTrackingFlashSAC/2026-09-12_15-01-19_mujoco \
      --clip walk_left90 --episodes 100
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
ap.add_argument('run_dir', help='训练 run 目录（含 policy.onnx）')
ap.add_argument('--clip', default='walk_left90', help='参考片段名（不含 .npz）')
ap.add_argument('--episodes', type=int, default=None,
                help='【旧语义，已弃用】凑够 N 个 episode 就停。会丢弃当时在跑的 episode，'
                     '造成右删失偏差（长 episode 更易被丢）。改用 --steps')
ap.add_argument('--steps', type=int, default=3000,
                help='跑固定 N 个控制步，收集窗口内【所有】完成的 episode。'
                     '消除旧 --episodes 的"凑够就停"删失偏差。默认 3000')
ap.add_argument('--num-envs', type=int, default=32,
                help='并行 env 数（默认 32：单 env 会严重低估 episode 长度）')
ap.add_argument('--seed', type=int, default=0, help='env RNG seed（保证可复现）')
ap.add_argument('--keep-dr', action='store_true',
                help='保留域随机化（默认关闭 —— 测干净策略能力，不被 DR 污染）')
ap.add_argument('--ctrl-dt', type=float, default=None,
                help='覆盖 ctrl_dt（秒）。用于测【策略决策频率下限】：'
                     '0.02=50Hz(训练值)，0.04=25Hz，0.08=12.5Hz，0.16=6.25Hz。'
                     '⚠️ 这不是"降频运行"，而是改变任务的时序语义（策略是在 0.02 下训的）')
ap.add_argument('--no-decouple', action='store_true',
                help='关掉参考运动解耦（默认【解耦】）。解耦=参考轨迹按墙钟实时播放，'
                     '只有策略决策频率随 ctrl_dt 变 —— 这才对应真实部署。'
                     '加此标志则参考运动随 ctrl_dt 一起变慢（0.04→半速），是混淆版本，仅供对照')
args = ap.parse_args()

RUN_DIR = Path(args.run_dir).resolve()
CLIP = args.clip
if not (RUN_DIR / 'policy.onnx').exists():
    print(f'❌ 需要 run 目录（含 policy.onnx）: {RUN_DIR}')
    raise SystemExit(1)

# ── flashsac 配置，task_name 对齐训练值 ──────────────────────────────────
with initialize(version_base=None, config_path='src/unilab/conf/flashsac'):
    cfg = compose(config_name='config', overrides=[
        'task=dm10_motion_tracking/base',
        'training.task_name=DM10MotionTrackingFlashSAC',
        f'env.commands.motion.params.motion_file=[motions/dm10/{CLIP}.npz]',
        'env.commands.motion.params.sampling_mode=uniform',
    ])

from unilab.base.config_adapter import create_env  # noqa: E402
from unilab.visualization.interactive_playback import (  # noqa: E402
    build_offpolicy_play_env_cfg_override,
)
from unilab.base.registry import ensure_registries  # noqa: E402

ensure_registries()
# ⚠️ env_cfg_override 是【替换】不是合并 → 必须以 play override 为底再加我们的键，
#    否则丢字段（如 max_episode_seconds）会在 validate 时报错。
override = dict(build_offpolicy_play_env_cfg_override('flashsac', cfg, root_dir=Path.cwd()) or {})
override['seed'] = args.seed
if not args.keep_dr:
    override['events'] = {}          # 关掉 DR：测干净能力
if args.ctrl_dt is not None:
    override['ctrl_dt'] = float(args.ctrl_dt)
    # ⚠️ ctrl_dt 一变，下面两个【派生量】跟着变（base.py 里都是 property，无需手动设）：
    #    sim_substeps      = round(ctrl_dt / sim_dt)       → 物理子步数
    #    max_episode_steps = int(max_episode_seconds/ctrl_dt) → episode 仍是 10s 墙钟
    #    这俩不整数/不为整时结果可疑，下面会打出来核验。

env = create_env(cfg, num_envs=args.num_envs, env_cfg_override=override)
robot = env.scene['robot']
cmd = env.command_manager.get_term('motion')
step_dt = float(env.step_dt)

# ── 参考运动解耦 ────────────────────────────────────────────────────────────
# MotionSampler.step() 是无条件 `current_frames += 1`（每控制步 1 帧，与 dt 无关，
# 见 motion_loader.py:591），片段 fps=50 → 训练 ctrl_dt=0.02 时 1/0.02=50 帧/s 正好实时。
# 若直接加大 ctrl_dt，参考运动会【跟着变慢】（0.04→半速、0.08→四分之一速），
# 等于同时改了「决策频率」和「参考速度」两个变量 —— 而速度本身会改变步态
# （memory: unilab-speed-reduction-improves-gait），测出来的东西答不了频率问题。
# 故 patch 成每控制步前进 round(ctrl_dt × fps) 帧：参考轨迹按墙钟实时，只变决策频率。
motion_fps = float(cmd.motion.fps)
frame_skip = max(1, int(round(step_dt * motion_fps)))
if args.no_decouple:
    frame_skip = 1                   # 混淆版本：参考运动跟 ctrl_dt 一起变慢
if frame_skip > 1:
    # ⚠️ 不能简单地"连调 N 次原 step"。终止判据 motion_clip_end 是
    #    `time_steps >= current_clip_end_frames`，且在帧自增【之前】求值
    #    （termination_manager 在 manager_based_rl_env.py:457 算，command_manager 在 :493）。
    #    frame_skip=2 时帧序列 …416 → 418 会【跳过】417 终点 → 终止永不触发 →
    #    帧号冲过数组末尾 → IndexError: index 418 out of bounds（实测踩到）。
    #    故必须夹住：一次前进 N 帧，但不超过 clip_end。
    def _sampler_step_skip(env_ids=None):
        s = cmd.sampler
        ids = np.arange(s.num_envs, dtype=np.int32) if env_ids is None else env_ids
        # 注意：current_frames[ids] 是花式索引 → 返回副本，必须用赋值写回，不能用 out=
        s.current_frames[ids] = np.minimum(
            s.current_frames[ids] + frame_skip, s.current_clip_end_frames[ids]
        )
        s._done_mask.fill(False)
        s._done_mask[ids] = s.current_frames[ids] > s.current_clip_end_frames[ids]
        return np.flatnonzero(s._done_mask)

    cmd.sampler.step = _sampler_step_skip

# ── 自检 ③：实际加载的片段 ─────────────────────────────────────────────────
# 脚本此前只回显 --clip 参数，那【不证明】override 生效了。这里读真实加载结果。
print(f'  片段自检: num_clips={cmd.motion.num_clips}  num_frames={cmd.motion.num_frames}'
      f'  clip_end={list(map(int, cmd.motion.clip_end_frames))}  fps={cmd.motion.fps:.0f}'
      f'  sampling_mode={cmd.sampler.mode}')

session = ort.InferenceSession(str(RUN_DIR / 'policy.onnx'), providers=['CPUExecutionProvider'])
in_name = session.get_inputs()[0].name


def policy_batch(obs: np.ndarray) -> np.ndarray:
    """ONNX 导出时 batch=1，逐 env 推理。"""
    return np.stack([session.run(None, {in_name: obs[i:i + 1]})[0][0]
                     for i in range(obs.shape[0])])


def get_obs(o):
    if hasattr(o, 'obs'):
        o = o.obs
    if isinstance(o, dict):
        return np.asarray(o.get('policy', list(o.values())[0]), dtype=np.float32)
    return np.asarray(o, dtype=np.float32)


bodies = list(robot.body_names)
li, ri, base = bodies.index('leg_l5_link'), bodies.index('leg_r5_link'), bodies.index('base_link')
termer = env.termination_manager
term_names = list(termer.active_terms) if hasattr(termer, 'active_terms') else []

n_ev = sum(len(v) for v in env.event_manager._mode_term_cfgs.values())
print(f'=== AB 测量 · {RUN_DIR.parent.parent.name}/{RUN_DIR.name} ===')
print(f'  片段 {CLIP} | num_envs {args.num_envs} | seed {args.seed} | DR term {n_ev} 个'
      f'{"（已关闭）" if n_ev == 0 else " ⚠️ 开启中"}')
_cfg = env._cfg if hasattr(env, '_cfg') else env.cfg
sim_dt = float(_cfg.sim_dt)
raw_decim = step_dt / sim_dt
mes = _cfg.max_episode_steps
print(f'  dt {step_dt}s | 步数预算 {args.steps} 控制步'
      f'{"（旧 --episodes=" + str(args.episodes) + " 已弃用）" if args.episodes else ""}')
print(f'  物理 {1/sim_dt:.0f}Hz (sim_dt={sim_dt})  ×  decimation {round(raw_decim)}'
      f'  →  策略 {1/step_dt:.2f}Hz (ctrl_dt={step_dt})  |  max_episode_steps {mes}')
if abs(raw_decim - round(raw_decim)) > 1e-6:
    print(f'  ⚠️ ctrl_dt/sim_dt = {raw_decim} 非整数 → decimation 被四舍五入为 {round(raw_decim)}，'
          f'实际 ctrl_dt = {round(raw_decim)*sim_dt}s（不是请求的 {step_dt}s）')
_playback = (frame_skip / step_dt) / motion_fps
if frame_skip > 1:
    _tag = '✅ 解耦：参考按墙钟实时'
elif step_dt > 1.0 / motion_fps + 1e-9:
    _tag = '⚠️ 未解耦：参考随 ctrl_dt 变慢（混淆版本）'
else:
    _tag = '（ctrl_dt=训练值，无需解耦）'
print(f'  参考 {motion_fps:.0f}fps | 每控制步前进 {frame_skip} 帧 → 播放 {_playback:.2f}x  {_tag}')
print()

N = args.num_envs
eps: list[dict] = []
censored: list[dict] = []
n_dropped = 0


def new_cur():
    return {'pos': [], 'ref': [], 'jt': [], 'rjt': [], 'qv': [], 'n': 0,
            'term': '?', 'f0': -1}


curs = [new_cur() for _ in range(N)]

state = env.reset()
if isinstance(state, tuple):
    state = state[0]
for i in range(N):                       # 初次 reset 后的起始帧
    curs[i]['f0'] = int(cmd.sampler.current_frames[i])

# 解耦实测：记录每控制步 current_frames 的【实际】增量（排除 reset 步，那一步会跳变）。
# 期望：全部 == frame_skip。若 ctrl_dt=0.04 时增量仍是 1，说明解耦没生效。
_frames_prev = np.asarray(cmd.sampler.current_frames).copy()
_fdelta: list[int] = []

steps = 0
while steps < args.steps:
    obs = get_obs(state)
    acts = policy_batch(obs)
    state = env.step(acts)
    if isinstance(state, tuple):
        state = state[0]
    steps += 1

    d = robot.data
    P = d.body_link_pos_w[:, [base, li, ri]]      # (N,3,3)
    R = cmd.body_pos_w[:, [base, li, ri]]
    J = d.joint_pos
    RJ = cmd.joint_pos
    Q = d.joint_vel
    done = np.asarray(state.terminated) | np.asarray(state.truncated)

    # 解耦实测：只统计【本步没有 reset】的 env（reset 会重抽起始帧，增量无意义）
    _fnow = np.asarray(cmd.sampler.current_frames)
    _alive = ~done
    if _alive.any():
        _fdelta.extend((_fnow[_alive] - _frames_prev[_alive]).tolist())
    _frames_prev = _fnow.copy()

    for i in range(N):
        c = curs[i]
        if not done[i]:
            c['pos'].append(P[i].copy())
            c['ref'].append(R[i].copy())
            c['jt'].append(J[i].copy())
            c['rjt'].append(RJ[i].copy())
            c['qv'].append(Q[i].copy())
            c['n'] += 1
            continue
        # ★ done 步：【整帧丢弃】。env.step() 内部已 autoreset（np_env.py:232-234，
        #   auto_reset 默认 True），此刻 robot.data 是【重置后】的位姿而非终止位姿。
        #   计入的话 episode 末尾会多一次"传送"，其 |Δpos|/step_dt 可达几十 m/s，
        #   且会被滑移/步态统计吃进去。代价：终止帧本身观测不到，长度少 1 步。
        fired = []
        for nm in term_names:
            try:
                if bool(np.asarray(termer.get_term(nm))[i]):
                    fired.append(nm)
            except Exception:
                pass
        c['term'] = '+'.join(fired) if fired else 'unknown'
        if c['n'] > 5:
            eps.append(c)
        else:
            n_dropped += 1               # 不再静默丢弃：计数并报告
        curs[i] = new_cur()
        curs[i]['f0'] = int(cmd.sampler.current_frames[i])   # reset 抽到的新起始帧

# 窗口结束时仍在跑的 episode 记为删失 —— 单独统计，不计入均值
for c in curs:
    if c['n'] > 5:
        c['term'] = 'censored'
        censored.append(c)

print(f'收集 {len(eps)} 个 episode（{steps} 控制步预算）'
      f' | 窗口末仍在跑(删失) {len(censored)} 个 | n<=5 丢弃 {n_dropped} 个')
if not eps:
    print('❌ 没收集到 episode')
    raise SystemExit(1)

# ── 自检 ①：起始帧分布 ──────────────────────────────────────────────────────
# ⚠️ 零假设依赖窗口长度：episode 长度 = num_frames − s，所以只有 s 足够大的
#    episode 才来得及在窗口内跑完。窗口 steps ≪ num_frames 时，能收上来的
#    episode 必然集中在高起始帧 —— 那是【删失的必然结果】，不是采样异常。
#    故仅当 steps ≥ 3×num_frames（删失占比小时）才把偏离均匀当成异常。
_sf = np.array([e['f0'] for e in eps if e['f0'] >= 0])
_nf = int(cmd.motion.num_frames)
_window_ok = steps >= 3 * _nf
if _sf.size:
    _edges = np.linspace(0, _nf, 9)
    _hist, _ = np.histogram(_sf, bins=_edges)
    _exp = _sf.size / 8
    print(f'  起始帧自检: n={_sf.size} min={_sf.min()} max={_sf.max()} 均值={_sf.mean():.0f}'
          f'（均匀理论均值 {(_nf-1)/2:.0f}）'
          f' | 窗口 {steps} 步 = {steps/_nf:.2f}× 片段长')
    print(f'    8 桶直方图 {_hist.tolist()}（均匀应各 ≈{_exp:.0f}）')
    if not _window_ok:
        print(f'    ⓘ 窗口不足 3×片段长，直方图必然偏高（只有高起始帧的 episode 跑得完）'
              f'—— 此窗口下【不判异常】，需加 --steps')
    elif _hist.min() < 0.4 * _exp or _hist.max() > 1.6 * _exp:
        print('    ⚠️ 窗口已足够长但仍明显偏离均匀 —— 采样分布与假设不符，先查清再往下')
    else:
        print('    ✅ 窗口足够长且分布接近均匀')

# ── 自检 ②：reset 传送是否已消除（每 episode 的最大单步位移）──────────────
_peak = []
for e in eps:
    p = np.asarray(e['pos'])
    if p.shape[0] > 1:
        _peak.append(float(np.linalg.norm(np.diff(p[:, :, :2], axis=0), axis=2).max()))
if _peak:
    _pk = np.array(_peak)
    print(f'  单步位移自检: 中位 {np.median(_pk)*1000:.1f} mm  最大 {_pk.max()*1000:.1f} mm'
          f'（若出现 ~米级 = reset 传送仍在污染）')

# ── 自检 ④：解耦实测（参考运动每控制步【实际】前进几帧）────────────────────
if _fdelta:
    _fd = np.array(_fdelta)
    _uniq, _cnt = np.unique(_fd, return_counts=True)
    _dist = '  '.join(f'{int(u)}帧×{int(c)}' for u, c in zip(_uniq, _cnt))
    _mode = int(_uniq[np.argmax(_cnt)])          # 主增量
    _clamped = int((_fd < frame_skip).sum())     # 被片段终点夹住的步数
    _obs_rate = float(_fd.mean())                # 帧/控制步（含夹住步，故略低于 frame_skip）
    _obs_play = (_obs_rate / step_dt) / motion_fps
    # 判据用【主增量】而非全体：夹住步是片段终点处的正常现象（frame_skip=2 时
    # 帧序 s,s+2,… 若 (clip_end−s) 为奇数就落在 clip_end−1 再被夹到 clip_end），
    # 每个 motion_clip_end episode 至多出现一次。若要求全体 == frame_skip 会误报。
    _ok = _mode == frame_skip
    print(f'  解耦实测: 主增量 {_mode} 帧/控制步（设定 frame_skip={frame_skip}）'
          f'  均值 {_obs_rate:.3f} → 播放 {_obs_play:.2f}x  {"✅" if _ok else "❌ 与设定不符"}')
    print(f'    增量分布: {_dist}（{_clamped} 步被片段终点夹住，属正常）')

fl = np.array([e['n'] for e in eps], dtype=float)


def agg(k):
    return np.concatenate([np.asarray(e[k]) for e in eps], axis=0)


pos, ref = agg('pos'), agg('ref')
jt, rjt, qv = agg('jt'), agg('rjt'), agg('qv')

print()
print('--- 判据 ① 跟踪质量 ---')
rel = np.linalg.norm((pos - pos[:, :1]) - (ref - ref[:, :1]), axis=2)
print(f'  左脚 平均位置误差 {rel[:, 1].mean():.4f} m')
print(f'  右脚 平均位置误差 {rel[:, 2].mean():.4f} m')
print(f'  关节角 平均误差   {np.abs(jt - rjt).mean():.4f} rad')

print()
print('--- 判据 ② 任务完成 ---')
print(f'  episode 均值 {fl.mean():.1f} 步 ({fl.mean()*step_dt:.2f}s)  中位 {np.median(fl):.0f}'
      f'  p10 {np.percentile(fl,10):.0f}  p90 {np.percentile(fl,90):.0f}')
from collections import Counter
for name, c in Counter(e['term'] for e in eps).most_common():
    print(f'    终止 {name}: {c}/{len(eps)} ({100*c/len(eps):.0f}%)')

print()
print('--- 判据 ③ 硬件负载 ---')
aq = np.abs(qv)
print(f'  |q̇| 峰值 {aq.max():.2f}  p95 {np.percentile(aq,95):.2f}'
      f'  p99 {np.percentile(aq,99):.2f}  均值 {aq.mean():.2f} rad/s')

print()
print('--- 附：步态（按 episode 内计算）---')
lifts, airs = [], []
slips_by_leg: dict[int, list[float]] = {1: [], 2: []}     # 1=左脚 2=右脚
for e in eps:
    p = np.asarray(e['pos'])
    for i in (1, 2):
        z = p[:, i, 2]
        zmin = float(np.percentile(z, 1))
        air = z > zmin + 0.02
        dxy = np.linalg.norm(np.diff(p[:, i, :2], axis=0), axis=1) / step_dt
        lifts.append(100 * (z.max() - zmin))
        airs.append(100 * air.mean())
        if (~air[:-1]).sum() > 2:
            slips_by_leg[i].append(float(dxy[~air[:-1]].mean()))
print(f'  抬脚 {np.mean(lifts):.1f} cm   腾空 {np.mean(airs):.1f}%')
# ⚠️ 分腿打印：单腿与两腿平均差得远（2026-09-09 同一次测量左 0.114 / 右 0.316，差 2.8 倍）。
#    只打合并值无法与历史单腿数字对话。该指标重尾，故一并给中位/p90。
for i, nm in ((1, '左脚'), (2, '右脚')):
    s = np.array(slips_by_leg[i])
    if s.size:
        print(f'  {nm} 接触滑移 {s.mean():.3f} m/s  中位 {np.median(s):.3f}'
              f'  p90 {np.percentile(s, 90):.3f}  (n={s.size})')
_pool = [v for v in slips_by_leg.values() if v]
if _pool:
    _all = np.concatenate([np.asarray(v) for v in _pool])
    print(f'  两腿合并 接触滑移 {_all.mean():.3f} m/s 中位 {np.median(_all):.3f} m/s')
