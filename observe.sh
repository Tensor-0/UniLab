#!/usr/bin/env bash
# ============================================================================
# observe.sh —— 一条命令拿到「这轮训练，硬件到底在干什么」
#
# 产出（→ ~/reports/dm10-observe/<时间戳>_<tag>/）：
#   summary.md    人读：分相预算 + 每档核利用率 + GPU
#   phases.json   机读：可用于历次 diff
#   percore.txt   每核明细（P / E / LP-E 分档）
#   gpu.csv       GPU 利用率 + 显存采样
#   train.log     原始训练日志
#
# 用法：
#   ./observe.sh --tag baseline                 # 默认 250 轮
#   ./observe.sh --tag with-compile --iters 250 --train-args "algo.algorithm.enable_compile=true"
#   ./observe.sh --diff <tagA> <tagB>           # 对比两次观测
#
# 为什么这样设计：
#   - 复用 train_safe.sh（CUDA 预检 + 禁止休眠）—— 否则 2026-09-18 那种
#     「静默降级到 CPU」的事故会污染整份画像
#   - 采样与训练并行，但采样器本身极轻（只读 /proc），不扰动测量
#   - phases.json 固定 schema ⇒ 可 diff；这是「长期能力」的关键
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ITERS=250
TAG="run"
TRAIN_ARGS="--algo ppo --task dm10_joystick_flat --sim mujoco"
SAMPLE_INTERVAL=0.25
OUTROOT="${HOME}/reports/dm10-observe"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iters)      ITERS="$2"; shift 2 ;;
        --tag)        TAG="$2"; shift 2 ;;
        --train-args) TRAIN_ARGS="$2"; shift 2 ;;
        --interval)   SAMPLE_INTERVAL="$2"; shift 2 ;;
        --diff)       DIFF_A="$2"; DIFF_B="$3"; shift 3 ;;
        -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
        *)            echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

# ---------- diff 模式 ----------
if [[ -n "${DIFF_A:-}" ]]; then
    A=$(ls -d "$OUTROOT"/*_"$DIFF_A" 2>/dev/null | tail -1)
    B=$(ls -d "$OUTROOT"/*_"$DIFF_B" 2>/dev/null | tail -1)
    [[ -n "$A" && -n "$B" ]] || { echo "找不到 tag: $DIFF_A 或 $DIFF_B" >&2; exit 1; }
    echo "对比：$(basename "$A")  vs  $(basename "$B")"
    python3 - "$A/phases.json" "$B/phases.json" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
def flat(d):
    out = {}
    for k, v in (d.get("scalars") or {}).items():
        if isinstance(v, dict) and "mean" in v: out[k] = v["mean"]
    for k, v in (d.get("summary") or {}).items():
        if isinstance(v, (int, float)): out[f"summary/{k}"] = float(v)
    return out
fa, fb = flat(a), flat(b)
keys = sorted(set(fa) & set(fb))
print(f"{'指标':<40s} {'A':>12s} {'B':>12s} {'差异':>10s}")
print('-'*78)
for k in keys:
    x, y = fa[k], fb[k]
    pct = (y/x - 1)*100 if x else float('nan')
    flag = '  <<<' if abs(pct) > 1.2 else ''   # 噪声底 ~1.2%
    print(f"{k:<40s} {x:>12.4f} {y:>12.4f} {pct:>9.1f}%{flag}")
print()
print("注：噪声底约 1.2%（同配置重跑实测）；小于它的差异不下结论。")
PYEOF
    exit 0
fi

# ---------- 观测模式 ----------
TS=$(date +%Y-%m-%d_%H%M%S)
DIR="$OUTROOT/${TS}_${TAG}"
mkdir -p "$DIR"
echo "观测目录: $DIR"

RUNS_BEFORE=$(ls -d logs/rsl_rl_ppo/DM10JoystickFlat/*/ 2>/dev/null | wc -l)

# 1) 启动训练（后台）
# shellcheck disable=SC2086
./train_safe.sh $TRAIN_ARGS algo.max_iterations="$ITERS" training.no_play=true \
    > "$DIR/train.log" 2>&1 &
TRAIN_BG=$!

# 2) 等训练进入稳态（跳过 import / chunk_tuner 冷启动）
sleep 20
if ! kill -0 "$TRAIN_BG" 2>/dev/null; then
    echo "⚠️ 训练在 20s 内就退出了，看 $DIR/train.log" >&2
    tail -20 "$DIR/train.log" >&2
    exit 1
fi

# 3) 采样：找真正的训练进程（train_rsl_rl.py），采 GPU + 每核
TPID=$(ps -eo pid,args 2>/dev/null | grep -F 'train_rsl_rl.py' | grep -v grep | awk '{print $1}' | head -1)
echo "训练进程 PID = ${TPID:-未找到}"

( for _ in $(seq 1 400); do
      nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu,temperature.memory,clocks_throttle_reasons.active \
          --format=csv,noheader,nounits 2>/dev/null
      sleep 0.5
  done > "$DIR/gpu.csv" ) &
GPU_BG=$!

uv run python scripts/observe/percore.py \
    --duration 45 --interval "$SAMPLE_INTERVAL" --timeseries \
    ${TPID:+--pid "$TPID"} --out "$DIR/percore.txt" 2>&1 | tail -1

kill "$GPU_BG" 2>/dev/null || true
wait "$TRAIN_BG" 2>/dev/null; TRAIN_RC=$?
kill "$GPU_BG" 2>/dev/null || true
echo "训练退出码=$TRAIN_RC"

# 4) 找这次的新 run，抽 Perf
NEW_RUN=$(ls -dt logs/rsl_rl_ppo/DM10JoystickFlat/*/ 2>/dev/null | head -1)
uv run python scripts/observe/collect_perf.py "$NEW_RUN" --out "$DIR/phases.json" 2>&1 | tail -1

# 5) 写 summary.md
python3 - "$DIR" "$TAG" "$ITERS" "$TRAIN_RC" <<'PYEOF'
import json, re, sys, statistics
from pathlib import Path
d, tag, iters, rc = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
ph = json.load(open(d/"phases.json")) if (d/"phases.json").exists() else {}
sc = ph.get("scalars", {})
cfg = ph.get("config", {})
sm = ph.get("summary", {})

L = []
L.append(f"# 观测报告 · {tag}\n")
L.append(f"| | |\n|---|---|")
L.append(f"| 时间 | {d.name} |")
L.append(f"| run | `{ph.get('run_name','?')}` |")
L.append(f"| device | **{cfg.get('device')}** |")
L.append(f"| num_envs / max_iter | {cfg.get('num_envs')} / {cfg.get('max_iterations')} |")
L.append(f"| 训练退出码 | {rc} |")
L.append(f"| 吞吐 | {sm.get('training_throughput_env_steps_per_sec')} env-steps/s |")
L.append(f"| 墙钟 | {sm.get('training_wall_time_sec')} s |")
L.append("")

c = sc.get("Perf/collection_time", {}).get("mean")
l = sc.get("Perf/learning_time", {}).get("mean")
fps = sc.get("Perf/total_fps", {}).get("mean")
if c is not None and l is not None:
    tot = c + l
    L.append("## 分相预算（轮均值）\n")
    L.append("| 阶段 | 秒/轮 | 占比 |\n|---|---|---|")
    L.append(f"| 采集 collection | {c:.4f} | **{c/tot*100:.1f}%** |")
    L.append(f"| 学习 learning | {l:.4f} | {l/tot*100:.1f}% |")
    L.append(f"| 合计 | {tot:.4f} | 100% |")
    L.append(f"\n`Perf/total_fps` = {fps:.0f}\n")

# per-core
pc = d/"percore.txt"
if pc.exists():
    t = pc.read_text()
    m = re.search(r"=== 每档利用率.*?===\n(.*?)\n\n", t, re.S)
    if m:
        L.append("## 每档核利用率\n")
        L.append("```")
        L.append(m.group(1).strip())
        L.append("```\n")
    m2 = re.search(r"最小 ([\d.]+) / 最大 ([\d.]+) / 均值 ([\d.]+) / 极差 ([\d.]+)", t)
    if m2:
        L.append(f"等价满载核数：最小 **{m2.group(1)}** / 最大 **{m2.group(2)}** / "
                 f"均值 **{m2.group(3)}** / 极差 {m2.group(4)}\n")
    mt = re.search(r"=== 进程 .*?每线程 CPU.*?===\n(.*?)\n\n", t, re.S)
    if mt:
        L.append("## 每线程 CPU（schedstat）\n")
        L.append("```")
        L.append(mt.group(1).strip()[:3000])
        L.append("```\n")
    mtemp = re.search(r"=== CPU 温度 ===\n(.*?)\n\n", t, re.S)
    if mtemp:
        L.append("## CPU 温度\n")
        L.append("```")
        L.append(mtemp.group(1).strip())
        L.append("```\n")
    mthr = re.search(r"=== 降频计数.*?===\n(.*?)\n\n", t, re.S)
    if mthr:
        L.append("## 降频计数\n")
        L.append("```")
        L.append(mthr.group(1).strip())
        L.append("```\n")

# gpu
gc = d/"gpu.csv"
if gc.exists():
    rows = [r.split(", ") for r in gc.read_text().splitlines() if r.strip()]
    def col(i):
        return [float(r[i]) for r in rows if len(r) > i and r[i].strip() not in ("", "[N/A]", "N/A")]
    u, mm = col(0), col(1)
    pw, tg, tm = col(3), col(4), col(5)
    if u:
        L.append("## GPU\n")
        L.append(f"- 利用率：min {min(u):.0f}% / mean **{sum(u)/len(u):.0f}%** / max {max(u):.0f}%")
        L.append(f"- 显存峰值：**{max(mm):.0f} MiB** / 8188")
        if pw:
            L.append(f"- 功耗：均值 **{sum(pw)/len(pw):.1f} W** / 峰值 {max(pw):.1f} W")
        if tg:
            L.append(f"- **温度**：核心 均值 **{sum(tg)/len(tg):.1f}°C** / 峰值 {max(tg):.0f}°C"
                     + (f" · 显存 均值 {sum(tm)/len(tm):.1f}°C / 峰值 {max(tm):.0f}°C" if tm else ""))
        thr = set(r[6].strip() for r in rows if len(r) > 6 and r[6].strip() not in ("", "0x0000000000000000", "Not Active"))
        L.append(f"- 降频原因：{('; '.join(sorted(thr)) if thr else '**无**')}\n")

(d/"summary.md").write_text("\n".join(L) + "\n")
print(f"已写入 {d/'summary.md'}")
PYEOF

# 6) 追加台账
INDEX="$OUTROOT/INDEX.md"
if [[ ! -f "$INDEX" ]]; then
    printf '# DM10 训练观测台账\n\n| 时间 | tag | device | 采集(s) | 学习(s) | fps | CPU均值核 | CPU峰值°C | GPU均值%% | GPU峰值°C | 备注 |\n|---|---|---|---|---|---|---|---|---|---|---|\n' > "$INDEX"
fi
python3 - "$DIR" "$INDEX" "$TAG" <<'PYEOF'
import json, re, sys
from pathlib import Path
d, idx, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ph = json.load(open(d/"phases.json")) if (d/"phases.json").exists() else {}
sc = ph.get("scalars", {})
def g(k): return sc.get(k, {}).get("mean")
pc = (d/"percore.txt").read_text() if (d/"percore.txt").exists() else ""
m = re.search(r"均值 ([\d.]+)", pc)
# CPU 温度峰值：取 package 那行
ct = '-'
mp = re.search(r"^\s*(Package id 0|Tctl)\s+(\S+)\s+(\S+)\s+(\S+)", pc, re.M)
if mp: ct = mp.group(4).rstrip('°')
gc = (d/"gpu.csv").read_text() if (d/"gpu.csv").exists() else ""
def col(i):
    out = []
    for r in gc.splitlines():
        p = r.split(", ")
        if len(p) > i and p[i].strip() not in ("", "N/A", "[N/A]"):
            try: out.append(float(p[i]))
            except ValueError: pass
    return out
u, gtemp = col(0), col(4)
gt = f"{max(gtemp):.0f}" if gtemp else "-"
row = (f"| {d.name.split('_')[0]} {d.name.split('_')[1]} | {tag} | {ph.get('config',{}).get('device')} | "
       f"{g('Perf/collection_time'):.4f} | {g('Perf/learning_time'):.4f} | {g('Perf/total_fps'):.0f} | "
       f"{m.group(1) if m else '-'} | {ct} | {sum(u)/len(u):.0f} | "
       f"{gt} |  |")
with idx.open("a") as fh:
    fh.write(row + "\n")
print(f"已追加台账 {idx}")
PYEOF

echo
echo "完成 → $DIR"
