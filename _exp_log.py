"""实验记录自动扫描：从 run_config.json 重建每次实验"改了什么"。

用法:
    uv run --no-sync python _exp_log.py              # 最近 10 个有效 run
    uv run --no-sync python _exp_log.py 20           # 最近 20 个
    uv run --no-sync python _exp_log.py --all        # 全部
    uv run --no-sync python _exp_log.py --run NAME   # 单个 run 详情

输出为 Markdown，可直接粘贴进 /home/zhan/reports/DM10步态_实验台账.md
"""
import json
import glob
import os
import sys

LOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'logs/rsl_rl_ppo/DM10JoystickFlat')
MIN_ITER = 100          # 低于此轮数的 run 视为冒烟测试，不计入实验
TRACK_FIELDS = ['sim_dt', 'ctrl_dt']


def load_runs():
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(LOG_ROOT, '2026-*', 'run_config.json'))):
        run_dir = os.path.dirname(cfg_path)
        name = os.path.basename(run_dir)
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        summary = {}
        sp = os.path.join(run_dir, 'run_summary.json')
        if os.path.exists(sp):
            try:
                summary = json.load(open(sp))
            except Exception:
                pass
        runs.append({'name': name, 'dir': run_dir, 'cfg': cfg, 'summary': summary})
    return runs


def snapshot(run):
    """提取用于对比的配置快照。"""
    cfg = run['cfg']
    reward = cfg.get('config', {}).get('reward', {}) or {}
    weights = {k: (v.get('weight') if isinstance(v, dict) else v) for k, v in reward.items()}
    env = cfg.get('config', {}).get('env', {}) or {}
    ranges = (env.get('commands', {}) or {}).get('twist', {}).get('ranges', {}) or {}
    misc = {f: env.get(f) for f in TRACK_FIELDS}
    return {'reward': weights, 'commands': ranges, 'misc': misc}


def diff_snapshots(prev, cur):
    """对比两个快照，返回 (配置变更列表, 命令变更列表)。"""
    out_cfg, out_cmd = [], []
    pr, cr = prev['reward'], cur['reward']
    for k in sorted(set(pr) | set(cr)):
        a, b = pr.get(k, None), cr.get(k, None)
        if a != b:
            out_cfg.append((k, a, b))
    pc, cc = prev['commands'], cur['commands']
    for k in sorted(set(pc) | set(cc)):
        a, b = pc.get(k, None), cc.get(k, None)
        if a != b:
            out_cmd.append((k, a, b))
    for k in TRACK_FIELDS:
        a, b = prev['misc'].get(k), cur['misc'].get(k)
        if a != b:
            out_cmd.append((k, a, b))
    return out_cfg, out_cmd


def fmt(v):
    if v is None:
        return '—'
    return str(v)


def print_run(run, prev=None, verbose=False):
    name = run['name']
    s = run['summary']
    it = s.get('completed_iterations', 0)
    rew = s.get('final_mean_reward')
    epl = s.get('mean_episode_length')

    head = f"### {name}"
    print(head)
    if it:
        print(f"- 轮数 {it}"
              + (f" ｜ reward {rew:.1f}" if isinstance(rew, (int, float)) else "")
              + (f" ｜ episode_len {epl:.0f}" if isinstance(epl, (int, float)) else ""))
    else:
        print("- (无 run_summary，可能未完成或被中断)")

    if prev is None:
        print("- 配置：基线（首个可比 run）")
    else:
        cfg_diff, cmd_diff = diff_snapshots(snapshot(prev), snapshot(run))
        if cfg_diff:
            print("- **奖励变更**：")
            for k, a, b in cfg_diff:
                print(f"    - `{k}`: {fmt(a)} → {fmt(b)}")
        if cmd_diff:
            print("- **命令/时步变更**：")
            for k, a, b in cmd_diff:
                print(f"    - `{k}`: {fmt(a)} → {fmt(b)}")
        if not cfg_diff and not cmd_diff:
            print("- 配置：与上一 run 相同")

    if verbose:
        snap = snapshot(run)
        print("- 全量奖励权重：")
        for k, v in sorted(snap['reward'].items()):
            print(f"    - `{k}` = {v}")
        print(f"- 命令范围: {snap['commands']}")
        print(f"- 时步: {snap['misc']}")
    print()


def main():
    args = sys.argv[1:]
    runs = load_runs()

    if args and args[0] == '--run':
        target = args[1]
        match = [r for r in runs if target in r['name']]
        if not match:
            print(f"未找到匹配 '{target}' 的 run")
            return 1
        idx = runs.index(match[-1])
        prev = runs[idx - 1] if idx > 0 else None
        print_run(match[-1], prev, verbose=True)
        return 0

    effective = [r for r in runs
                 if r['summary'].get('completed_iterations', 0) >= MIN_ITER]

    if args and args[0] == '--all':
        limit = len(effective)
    else:
        limit = int(args[0]) if args and args[0].isdigit() else 10

    shown = effective[-limit:]
    print(f"# DM10 实验记录（最近 {len(shown)} / 有效实验 {len(effective)} / 全部 run {len(runs)}）\n")
    for i, r in enumerate(shown):
        # 与上一个「有效实验」对比，跳过中间冒烟测试 run
        prev = shown[i - 1] if i > 0 else None
        print_run(r, prev)
    return 0


if __name__ == '__main__':
    sys.exit(main())
