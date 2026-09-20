#!/usr/bin/env bash
# ============================================================================
# train_safe.sh —— 训练安全包装器（2026-09-19 建立）
#
# 【它解决什么】两个已用日志确证的静默故障：
#
#   事故还原（2026-09-18）：
#     01:07:21  NetworkManager: sleep requested
#     01:07:28  systemd-sleep: Entering sleep state 'suspend' (s2idle)
#               ↑ 三条并行训练在此被【系统休眠】杀死（不是 CUDA 崩溃）
#     01:20:13  唤醒
#     01:20:16  NVRM: Xid 31, python3, MMU Fault   ← GPU 未正常恢复
#     01:24:25  启动新训练
#               → torch.cuda.is_available() = False
#               → get_default_device() 静默返回 "cpu"（零警告）
#               → 整场 3000 轮跑在 CPU 上，55 分钟而非 ~24 分钟
#
#   根因代码路径：
#     conf 里 training.device = null
#       → train_rsl_rl.py:460  default_device=get_default_device()
#       → uni_rl/algos/rsl_rl.py:64  `configured_device or default_device`
#       → src/unilab/utils/device.py:18  cuda 不可用 ⇒ 静默 return "cpu"
#   注意 resolve_torch_device_alias() 那个"绝不静默降级"的校验函数，
#   在训练路径上【零调用】——承诺只写在 docstring 里。
#
# 【本脚本做两件事】
#   ① 启动前校验 CUDA 可见；不可见就【中止】，绝不静默跑 CPU
#   ② 训练期间用 systemd-inhibit 禁止系统休眠
#
# 【用法】替代 `uv run train`：
#   ./train_safe.sh --algo ppo --task dm10_joystick_flat --sim mujoco \
#       algo.max_iterations=3000
#
#   其余参数原样透传给 `uv run train`。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- ① 启动前校验 CUDA（这一步就是为了让静默失败变成响亮失败）----
if ! uv run python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    cat >&2 <<'MSG'
❌ CUDA 不可见 —— 训练会静默降级到 CPU（实测慢 2.3 倍）。已中止。

   最常见原因：刚从系统休眠唤醒，NVIDIA 驱动没恢复（日志里的 Xid 31）。
   处置（由轻到重）：
     1) 先看 nvidia-smi 能否列出 RTX 4060
     2) 重载驱动：sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia
     3) 最稳妥：重启

   参考：全历史 60+ 条 run 里唯一一条 CPU run，就是休眠唤醒后 4 分钟启动的那条。
MSG
    exit 1
fi

# ---- ② 训练期间禁止系统休眠 ----
# --what=idle 挡"空闲自动挂起"，sleep 挡真正的 suspend。
echo "✅ CUDA 可见；训练期间将禁止系统休眠（systemd-inhibit）"
exec systemd-inhibit \
    --what=idle:sleep \
    --why="UniLab RL training in progress (train_safe.sh)" \
    --mode=block \
    uv run train "$@"
