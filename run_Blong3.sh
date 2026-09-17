#!/bin/bash
# AB 实验 · B 组长训练（奖励最小化组）
# 创建于 2026-09-12，重启后重建（原 /tmp/ab_test/run_Blong2.sh 已随重启丢失）
# A 组对照：logs/AB_A_long（2026-09-12 16:01 完成，25000 iter）
mkdir -p /home/zhan/UniLab/ab_test
cd /home/zhan/UniLab
uv run --no-sync train --algo flashsac --task dm10_motion_tracking --sim mujoco \
  algo.max_iterations=25000 \
  algo.algo_log_name=AB_B_long3 \
  algo.save_interval=2500 \
  reward.action_rate_l2=null \
  reward.joint_limit=null \
  reward.undesired_contacts=null \
  reward.motion_joint_pos=null \
  reward.motion_joint_vel=null \
  > /home/zhan/UniLab/ab_test/B_long3.log 2>&1
echo "B_LONG3_EXIT=$?" >> /home/zhan/UniLab/ab_test/B_long3.log
