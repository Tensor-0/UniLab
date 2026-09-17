#!/usr/bin/env python3
"""渲染指定 run 的 checkpoint 回放视频（绕过 CLI 的保留字校验）。

用途：AB 对照实验 —— 用完全相同的渲染参数分别出 A/B 组视频。
CLI 走不通的原因：training.play_only 是 RESERVED_OVERRIDE_KEYS，且
--load-run 只接受 run 目录名（不接受路径），而 eval 子命令又没有
--algo-log-name 参数。

做法：从 run_config.json 恢复 cfg（与训练时逐字一致），把 algo_log_name
直接设为 <abs_log_root>/<task>，使 resolve_checkpoint_path 的
base_dir/load_run 拼接命中真实目录。

用法:
  python render_run.py <run_dir> [--steps N]
  例: python render_run.py /home/zhan/UniLab/logs/AB_A_long/DM10MotionTrackingFlashSAC/2026-09-12_15-01-19_mujoco
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run 目录（含 model_*.pt 与 run_config.json）")
    ap.add_argument("--steps", type=int, default=None, help="回放步数（默认用配置值 800）")
    ap.add_argument("--algo", default="flashsac")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"❌ 不是目录: {run_dir}")
        return 1

    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        print(f"❌ 缺 run_config.json: {config_path}")
        return 1

    ckpts = sorted(run_dir.glob("model_*.pt"))
    if not ckpts:
        print(f"❌ {run_dir} 下没有 model_*.pt")
        return 1

    # run_config.json 是训练时的完整快照 —— 用它保证渲染配置与训练一致
    cfg = OmegaConf.create(OmegaConf.load(config_path))

    # 关键：让 resolve_checkpoint_path 的 (base_dir / load_run) 直接命中真实目录。
    # 源码 checkpoint.py:  base_dir=root_dir/"logs"/algo_log_name, run_dir=base_dir/load_run
    # 取 algo_log_name=<abs>/AB_A_long/DM10MotionTrackingFlashSAC 且 load_run=<时间戳目录名>
    base_dir = run_dir.parent                     # .../AB_A_long/DM10MotionTrackingFlashSAC
    relative = base_dir.relative_to(Path.cwd() / "logs")   # AB_A_long/DM10MotionTrackingFlashSAC
    OmegaConf.update(cfg, "algo.algo_log_name", str(relative), merge=False)
    OmegaConf.update(cfg, "algo.load_run", run_dir.name, merge=False)
    OmegaConf.update(cfg, "training.play_only", True, merge=False)
    if args.steps is not None:
        OmegaConf.update(cfg, "training.play_steps", args.steps, merge=False)

    # ⚠️ run_config.json 只记录训练时被显式覆盖的值；play_render_mode /
    # play_env_num / play_steps / cam_* 等是 Hydra 默认值，快照里可能不存在。
    # 缺失时补上 conf 里的默认值，避免 play_offpolicy 读到空 key。
    PLAY_DEFAULTS = {
        "training.play_render_mode": "auto",
        "training.play_env_num": 16,
        "training.play_steps": 800,
        "training.export_onnx": True,
        "training.cam_distance": 6.0,
        "training.cam_elevation": -20.0,
        "training.cam_azimuth": 90.0,
        "training.no_play": False,
    }
    for key, default in PLAY_DEFAULTS.items():
        if OmegaConf.select(cfg, key) is None:
            OmegaConf.update(cfg, key, default, merge=False)
            print(f"  [补默认值] {key} = {default}")

    print(f"run_dir   : {run_dir}")
    print(f"checkpoint: {ckpts[-1].name}")
    print(f"algo_log_name={cfg.algo.algo_log_name!r}  load_run={cfg.algo.load_run!r}")
    print(f"render    : mode={cfg.training.play_render_mode} steps={cfg.training.play_steps} "
          f"envs={cfg.training.play_env_num}")

    sys.path.insert(0, str(Path.cwd() / "src"))
    from unilab.scripts.train_offpolicy import play_offpolicy

    video = play_offpolicy(args.algo, cfg)
    print(f"\n{'✅ 视频: ' + str(video) if video else '❌ 未生成视频'}")
    return 0 if video else 1


if __name__ == "__main__":
    raise SystemExit(main())
