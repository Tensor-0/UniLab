# DM10 分支说明

> `dm10` 分支 = 在上游 UniLab 基础上做 **达妙 DM10 双足人形（10 自由度下肢）** 的全部工作。
> 本文是**总览**：这个分支有什么、现在到哪了、各文档在哪。
>
> 最后更新：2026-09-21

## 一句话

**一台 10 自由度双足人形下肢（DM10），用 RL 学会走路。**

```
机器人   10 DoF：两腿各 5 关节（髋P / 髋R / 髋Y / 膝 / 踝）
电机     10× 达妙 DM-J4340-2EC（24V，减速比 40，固件力矩上限 28 N·m）
实机质量 6.12 kg（含板卡/IMU/线缆）—— 12 个 body 全部实测
训练     UniLab + MuJoCo（物理跑 CPU）+ PPO
```

---

## 一、这个分支有什么

### 1.1 任务（`src/unilab/conf/`）

| 任务 | 说明 |
|---|---|
| `ppo/task/dm10_joystick_flat` | ⭐ **主线** —— 听速度口令走路（39 维观测）|
| `ppo/task/dm10_motion_tracking` | 跟着参考动作走 |
| `flashsac/task/dm10_motion_tracking` | 同上，换算法 |
| `flashsac/task/dm10_actuator_motion_tracking` | 同上 + 真实执行器模型 |

### 1.2 模型资产（`src/unilab/assets/robots/dm10/`）

⚠️ **4 个模型文件，是「质量」×「执行器」两个维度的组合**：

| 文件 | 质量 | 执行器 |
|---|---|---|
| `dm10.xml` | 估的 10.86 kg | `<position>` 理想伺服 |
| `dm10_actuator.xml` | 估的 10.86 kg | `<dcmotor>` 真实 τ-ω |
| `dm10_measured.xml` | **实测 6.12 kg** | `<position>` |
| ⭐ `dm10_measured_actuator.xml` | **实测 6.12 kg** | **`<dcmotor>`** |

配套 scene 各一个（`scene_flat*.xml`），必须 **lockstep** ——
`scene_flat_actuator.xml` 的注释明确警告：不镜像的改动会**静默多出一个实验变量**。

另有一组**实验资产**（不是主线，勿误用）：
`dm10_sens_{m1,m005,p005,p1}.xml` + `scene_sens_*.xml` —— 执行器参数灵敏度扫描用。

### 1.3 工具

| 工具 | 干什么 |
|---|---|
| `train_safe.sh` | 训练包装：CUDA 预检（防静默掉 CPU）+ `systemd-inhibit` 防休眠 |
| `observe.sh` + `scripts/observe/` | 一条命令出硬件画像（每核分档 / 每线程 schedstat / 温度 / 降频）|

### 1.4 文档

| 文件 | 定位 |
|---|---|
| `HANDOFF-dm10-actuator.md` | 执行器建模线（τ-ω 曲线、`dcmotor`、飞轮）|
| `HANDOFF-dm10-calibration.md` | 真机标定线（电机 ID / CAN / 零点 / motor_sign）|
| `HANDOFF-unisim-1.7.2-migration.md` | 依赖迁移（含回滚步骤）|
| `docs/dm10/README.md` | 结构图 + 实测质量对照 |
| **本文** | 总览（入口）|

---

## 二、现在到哪了

### 2.1 已完成

| 项 | 结果 |
|---|---|
| **真机跑通** | ✅ 三层对照证实软件链路正确（真机 obs → 同一个 onnx → 平均差 4.8°）|
| **模型↔真机对齐** | ✅ 五层实测完成（电机 ID 反序 / CAN / 零点 / IMU / motor_sign）|
| **模型质量修正** | ✅ 10.86 → **6.12 kg**（12 个 body 全部实测，三个独立数字互印证）|
| **真实执行器接入** | ✅ 10 个电机的 τ-ω 曲线（`<dcmotor>`）|
| **PD 站立保底** | ✅ 异常时切 PD 而非瘫软 |
| **训练提速** | ✅ **3.7×**（55 → 14.7 分钟）：修 learner 掉 CPU 2.44× + 换 Newton 求解器 1.64× |
| **依赖升级** | ✅ unisim-core 1.1.4 → 1.7.2（env 步进 **+14.2%**）|

### 2.2 待办（按优先级）

| # | 事项 | 说明 |
|---|---|---|
| **1** | ⭐ **用新模型跑一版训练** | `dm10_measured_actuator.xml` **从没被训练用过** —— 先确认它可用 |
| **2** | **试「刚度退火」** | 调研认为这才是该做的（不是调 kp）：`K0=60 → K*=40`，机理=硬关节扩大 viability kernel。CoRL 2026 有 Spot 的结果，**本机未复现** |
| 3 | 步态三个待解数值 | 抬脚不对称 19.8/8.2 cm、步频 3.85 Hz（目标 1.5）、速度超指令 36%。**只有 1 seed，需补 2~3 条** |
| 4 | 补 L0（站立 + 抗扰） | 现在不是"不会站"，是训练里 `rel_standing_envs: 0.0` ⇒ 策略**从没见过零命令** |
| 5 | 粗糙地形 | 本机框架下只要 2 个文件（共享 hfield）|
| 6 | 视觉（L3/L4）| 本框架**没有相机观测**，要换栈（mjlab 或 IsaacLab）|

### 2.3 已排除的方向（别再试）

| 方向 | 为什么排除 |
|---|---|
| 多 run 并发 | UniLab #1328：物理相 ~16 核就饱和；tdmpc 作者：会变慢 |
| APPO / 采集-学习 overlap | 天花板 1.19×，消费级机器实测打平 |
| `torch.compile` | rsl_rl PR #199：**MLP 没收益** |
| 扫 kp 看谁能站住 | MuJoCo 里 kp 到 30000 都不振；文献说 RL 各档增益都能训 |
| ±20% 增益随机化 | `arXiv:2011.02404`：**证明没用** |
| 给 `dcmotor` 加力矩上限 | `ctrlrange` 已经是电压上限（MuJoCo #3489）|

---

## 三、几个必须知道的事实（踩过的坑）

### 3.1 模型质量：不是统一偏重，是每段都错

```
仿真的 10.86 kg 是【照抄达妙官方 CAD】，不是编的
但官方那个是「含双臂」的 13.14 kg 去掉双臂 = 10.86
而实机只有 6.12 kg —— 差在【打印件密度】被按实心塑料估了
```
逐段误差 **0.87× ~ 3.86×**，横跨 4.4 倍 ⇒ **乘一个系数修不了，必须逐段重建**。

### 3.2 10 个电机的分布（从实测质量反推 + 用户核对）

```
loin_yaw 2 个 | l1 1 个 | l2 【0 个】| l3 【2 个】| l4 1 个 | l5 【0 个】
合计 2 + (1+0+2+1+0)×2 = 10  ✅
```
**判据**：按"每段各一个"算，`leg_l2`(54.3 g) 和 `leg_l5`(64.1 g) 会得到**负的打印件质量**。

### 3.3 PD 增益：质量改了不用调

三条独立论证：**De Luca 下界 `α=m·g·d` ∝ 质量** / 关节有效惯量 **88% 来自电机** /
主流公式（ANYmal、宇树 2026）**都不含本体质量**。

本机实测：髋R 的 α `46.78 → 28.26 N·m`，kp=30 余量 **0.64× → 1.06×**
（质量修正反而让它从"不够"变成"刚好够"）。

⭐ 而「仿真 kp=30 / 真机 16-20」的不一致是**必要的**（PD 语义不同：
MuJoCo `<position>` 是隐式约束 vs 达妙 MIT 是固件离散 PD），**桥接靠刚度退火**。

### 3.4 训练必须用 `train_safe.sh`

2026-09-18 踩过：系统进 s2idle 休眠 → 唤醒后 CUDA 不可见 →
`get_default_device()` **静默返回 cpu** → 整场慢 2.3 倍、零报错。

---

## 四、怎么跑

```bash
# 训练（主线任务）
cd ~/UniLab
./train_safe.sh --algo ppo --task dm10_joystick_flat --sim mujoco

# 用【实测质量 + 真实执行器】的模型（需先在 owner yaml 里指向新 scene）
#   scene_flat_measured_actuator.xml

# 硬件画像（一条命令）
./observe.sh --tag baseline

# 回放
uv run eval --algo ppo --task dm10_joystick_flat --sim mujoco --load-run -1
```

**查看模型结构**：
```bash
# Studio（中文界面，见 mujoco-zh）
DISPLAY=:1 PYTHONPATH=~/mujoco-zh/src ~/UniLab/.venv/bin/python -m mujoco_zh.panel_zh \
  ~/UniLab/src/unilab/assets/robots/dm10/scene_flat_measured_actuator.xml
```

---

## 五、相关仓库

| 仓库 | 内容 |
|---|---|
| `Tensor-0/UniLab`（本仓，`dm10` 分支）| 训练 + 模型 + 配置 |
| `Tensor-0/unilab-perf` | 性能观测工具 + 实测结论 + 原始数据 |
| `Tensor-0/mujoco-zh` | MuJoCo Studio 中文化（234 条界面文字）|
| `Tensor-0/robot-deploy-toolkit` | 真机部署工具链（8 阶段 + 7 脚本）|
| `Tensor-0/dm10-motion-tracking`（私有）| 参考动作转换 |

⚠️ **上游是 `Motphys/UniLab`**（本仓是它的 fork）。`dm10` 分支的开发**不回推上游**。
