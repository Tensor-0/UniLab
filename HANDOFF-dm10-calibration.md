# DM10 交接 — 2026-09-18（⭐ 根因已修复，策略真的会走了）

> ## ⭐ 接续点（compact 后先读这六条）
>
> 1. **⭐⭐⭐ 根因已修复并验证**（见 §8.9 + §10）：
>    `feet_air_time` 的命令门控**恒 False**（`_command_gate` 用**严格大于**
>    `total > 0.5`，而命令采样上界恰好 0.5）⇒ 唯一奖励「脚离地」的项从未生效。
>    **改 `command_threshold: 0.5 → 0.05` 后重训 ⇒ 腾空 0% → 50%，视频确认真的在走。**
>    **⇒ 一个数字的修复。**
> 2. **⭐ 新策略已产出、可直接部署**（见 §10.2）：
>    `logs/rsl_rl_ppo/DM10JoystickFlat/2026-09-18_01-24-25_mujoco/`
>    （`policy.pt` + `policy.onnx` 均在，onnx 大小与板上那份一致）
> 3. **⚠️ 三个待解**（见 §10.4）：左右**不对称**（抬脚 19.8 vs 8.2 cm）、
>    步频 **3.85 Hz 偏快**（目标 1.5）、速度**超指令 36%**
> 4. **⚠️ 只有 1 seed × 1 episode ⇒ 不能下最终结论**（memory 教训：
>    单 seed 下结论我犯过，4 seed 后翻案）
> 5. **✅ 软件链路正确**（真机 obs → 同一个 onnx → 平均差 4.8°）；
>    **⚠️ 两条旧结论已撤回**（§8.5.1 幅度 3~8 倍 / §8.5.2 IMU 零偏 18°）
> 6. **⚠️ 物理参数从未更新**（见 §10.5）：仿真质量 10863 g，实机约 5200 g，**差一倍**
>
> **已完成**：五层标定 / `motor_sign` 三轮验证 / 限位 10/10 / 39 维部署就绪 /
> PD 站立保底 / TX 假死检测 / 真机跑策略 + 三层对照 / **根因定位 + 修复 + 重训验证**

---

## 一、今天最终成果

| 层 | 结果 | 手段 |
|---|---|---|
| 总线 ↔ 左右腿 | **can1 = 左、can2 = 右** | 只动一条腿差分两总线 |
| CAN ID ↔ 关节 | **踝=1、膝=2、髋roll=3、髋yaw=4、髋pitch=5**（倒序）| 逐关节 5/5 |
| 电机零点 | **10 台全点零，残留 0.01°** | `probe_set_zero_all.py` |
| IMU 三轴 | **X前 / Z上 / 转身轴**，倾角 ≈0° | 三次采样 |
| ⭐ **motor_sign** | **`[+1,-1,-1,+1,+1,-1,-1,-1,-1,-1]`** | 逐台目视，10/10 + 二轮复核 |

### 配置已写入

`roboparty_deploy/src/inference/robots/dm10/robot.yaml:74`
```yaml
motor_sign: [ 1, -1, -1, 1, 1, -1, -1, -1, -1, -1]
#             l1  l2  l3  l4 l5  r1  r2  r3  r4  r5
#            髋P 髋R 髋Y  膝 踝  髋P 髋R 髋Y  膝  踝
```
**改前备份**：`robot.yaml.bak-motorsign-20260917-194750`（最初）、`robot.yaml.bak-motorsign2-20260917-203736`（第二次修正前）

### ⭐ 规律（不是随机，跟着轴的几何走）

| 关节类型 | 左 | 右 | 原因 |
|---|---|---|---|
| **膝 / 髋pitch / 踝** | **+1** | **−1** | 左右**镜像装配**（在腿的前后平面内）|
| **髋roll / 髋yaw** | **−1** | **−1** | 左右**同向装配**（同向轴）|

**⚠️ 我一度误判为"左腿内部不一致"** —— 实际是把"左右镜像"理解错了：
**右腿往外张 和 左腿往内收是【同一个物理方向】。**

---

## 二、⭐ 今天犯的 5 个错误（比结论值钱）

### 1. 自写解码按【小端】⇒ 造出两个假故障
我解出 `-12.35 rad`，据此断言"位置反馈坏了""点零破坏标定" —— **全错**。
**官方手册 V1.3 p.13 明确：位置是大端 `(d[1]<<8)|d[2]`，`0x8000`=0 rad。**
驱动 `dm_motor_driver.cpp:235` 写的正是大端 —— **驱动是对的，我错了。**
> **教训**：自写解析器必须先用手册/官方实现交叉验证，再用来下结论。

### 2. `get_error_id()` 恒返回 0 是【驱动 bug】，不是"没使能"
`dm_motor_driver.cpp:238` 只在高4位 >7 时写 `error_id_`，而 **err=1(使能) 不>7** ⇒ 永远保持初值 0。
**连带 `init_motor()` 返回值也不可信**（它的 `switch` 命中 `DM_DOWN=0`）。
**正解**：自己解反馈帧 `d[0]>>4`。见 memory `dm-driver-error-id-bug`。

### 3. ⭐ 使能时序错 —— 用户 2026-09-04 已实测并提交在 git 里
`dm-dual-motor-test` 提交 `8e64d09`：
> 切控制模式会清除使能状态，原流程（先使能后切模式）导致电机从未真正使能，**反馈状态≠1 静默被忽略**。

**正解**：`unlock → set_motor_control_mode(MIT)【失能态下】→ lock → 保持2s 并校验 err==1`
**我错在哪**：调了 `init_motor()` 之后又调 `set_motor_control_mode`，且没校验使能。

### 4. 增益是自己拍的，而官方值就在仓库文档里
我用 kp=8 → 30。**而 `docs/03-单关节稳停/0-index.md` 早写了官方值 `kp=16~20 / kd=3`。**
文档还点名了我的错误做法：
> 如果某个关节的 kp 需要调到和别的关节差很多才能稳，那多半不是 kp 的问题。

### 5. ⭐ 幅度太小 —— 用户两次纠正
关节有 **~0.58 N·m 的恒定阻力**，`kp20 × 0.03 = 0.6 N·m` **刚好卡在平衡点** ⇒ 位置几乎不动。
**加大【幅度】比加大 kp 好**（kp 保持官方值可避开振荡）。
0.03 → 0.15 后，位移从 0.0023 涨到 **0.042（18 倍）**。

---

## 三、当前状态速查

| 项 | 值 |
|---|---|
| 板子 IP | **`<BOARD_IP>`**（会漂；认不出按 MAC `<BOARD_MAC>` 扫 ARP）|
<!-- 真实值已脱敏，本地保留在 /tmp/HANDOFF-dm10-calibration.md.UNREDACTED -->
| SSH | `<USER>@<BOARD_IP>` |
| 机器人 | 吊架悬挂、腿悬空、10 台电机装好 |
| CAN | **can0/can1** 都 `<FD> ERROR-ACTIVE`、无 loopback（⚠️ 2026-09-17 晚从 can1/can2 改来，见下）|
| 零位姿态 | **直腿 = qpos=0** |

### ⚠️⚠️ CAN 接口名 — 2026-09-17 晚的重大变更（读这段再动手）

**接口名从 `can1`/`can2` 改成了 `can0`/`can1`，且做了三处修复：**

| # | 改了什么 | 为什么 |
|---|---|---|
| 1 | `robot.yaml`: `motor_interface: ["can1","can2"]` → **`["can0","can1"]`** | 官方手册（DM-USB2CANFD_Dual V1.0 p.12-13）与 `DEPLOY_STATUS.md` 的设计**都是 can0/can1** |
| 2 | **禁用板载 CAN**（`systemctl disable can4.service` + 注释 `98-onboard-can.rules`）| ⚠️ **板载 spi5.0 启动时先抢 `can0` 号**（dmesg: `spi5.0 can4: renamed from can0`），若 USB 适配器在释放前枚举，就只能拿 can1/can2 ⇒ **名字会漂**。禁用后 can0 号不再被抢 |
| 3 | `probe_direction.py` 的 `ORDER` 表：**写死的 can1/can2 → 从 robot.yaml 读** | ⚠️ 写死的表在接口改名后给出错误提示（"配置里没有 can0 ID=2"），**险些让我误判"配置反了"** |

**备份**：
- `robot.yaml.bak-caniface-<时间戳>`
- `/etc/systemd/system/can4.service.bak-20260917-221224`
- `/etc/udev/rules.d/98-onboard-can.rules.bak-20260917-221224`

**⭐ 哪条通道是哪条腿 —— 端到端实测（走生产路径：从 robot.yaml 读 motor_sign 并乘进下发值）：**
```
给 can0 ID=2（配置认为左膝）→ 操作员确认：左腿  ✅
给 can1 ID=2（配置认为右膝）→ 操作员确认：右腿  ✅
⇒ motor_interface: ["can0","can1"] 正确（入口0=左、入口1=右）
```

**⚠️ 若以后换 USB 口或改接法 ⇒ 必须重做这个验证**（"哪条接口是哪条腿"不是软件能自动判断的）。

**物理侧**：模块 CH1 → can0、CH2 → can1（与官方手册一致）。适配器现插在 USB 口 `1-1.2`。

### 常用命令
```bash
ssh <USER>@<BOARD_IP>
cd ~/roboparty_deploy && source /opt/ros/humble/setup.bash && source install/setup.bash

cd ~/robot-deploy-toolkit
# 只读全状态
python3 scripts/sample_motors.py --config ~/roboparty_deploy/src/inference/robots/dm10/robot.yaml
# ⭐ 单个关节方向测试（会动！）
python3 scripts/probe_direction.py --bus can2 --motor-id 2 --amplitude 0.15 --one-way 1 --hold-view 20 --confirm
# 只验证使能时序
python3 scripts/probe_direction.py --bus can2 --motor-id 2 --enable-check --confirm
```

---

## 四、⬜ 下一步（按优先级）

| # | 事项 | 为什么 |
|---|---|---|
| **1** | ~~上机前单关节复核~~ | ✅ **已完成 10/10**（2026-09-17 晚）—— 用 `probe_direction.py --use-config-sign`，**从 robot.yaml 读 motor_sign 乘进下发值 ⇒ 走完整生产路径**（前三轮都绕过配置，配置写错也测不出来）。记录 `results/preflight_singlejoint_20260917.json` |
| **2** | ~~复核左髋yaw~~ | ✅ **已完成** —— 用「左右配对对比」复核（两腿给同样正角，观察是否同侧）。⚠️ **教训：判据要用物理描述，不能用角度约定**（转角换算本轮错了 3 次） |
| **3** | ~~关节限位检查~~ | ✅ **已完成 10/10 通过**（2026-09-17，v2）。⚠️ 两个坑：① **别用 `scripts/limit_check.py`**（已加拒绝守卫）—— kp=0 纯阻尼实测关节根本不动却报'✓'（假成功）；② **v1 没乘 `motor_sign`** ⇒ 测的方向是反的。**正确语义：电机指令 = 关节角目标 × motor_sign**。记录 `results/limit_check_20260917.json` |
| **4** | ~~修 `joint_limits` 索引错位~~ | ❌ **已撤回 —— 那不是 bug**。上游作者主动在提交 `7f1769ec`「fix index bug」里去掉过 `usd2urdf_`；我误判为"缺重排"并改了，**已回滚**。详见 memory `dm-driver-error-id-bug` 撤回段 |
| **5** | 质量/尺寸实测 | `DM10下肢_实机参数登记表.md` 12 个 body 仍是仿真猜测值 |

### ⭐ 有效的复核方法（本轮实测得出的）
**「左右配对对比」** —— 两个镜像/同向的关节**给同样的正角命令**，观察是否**同侧/相反侧**：
```
髋pitch/膝/踝 → 模型要求【镜像】⇒ 应该看到【相反侧】
髋roll/髋yaw  → 模型要求【同向】⇒ 应该看到【同侧】
```
**⇒ 操作员只需比较两侧，不需要判断单侧方向、不需要任何角度换算。**
**⇒ 而且能同时交叉检验：`motor_sign` 的左右关系 应当与 模型的镜像/同向 一致。**

### ❌ 试过但**不可行**的方法（别再花时间）
**IMU 测机体反作用**：让髋yaw 转，用 IMU 读机体 wz 推方向。
**实测失败** —— 腿的转动惯量远小于机体，反作用弱；机体还被吊架约束。
读数淹没在噪声里（三个方向全在 0.001~0.002 rad/s，与静止 -0.0003 同量级）。

### ⚠️⚠️ 最难的一条教训
**判据必须用【物理描述】（往屁股/往前、伸/收、压/抬、外张/内收），
不能用【角度约定】（"俯视顺时针"）。**
**同类坑还有一个：凡是【关节角 ↔ 电机指令】的换算，必须显式过 `motor_sign`** ——
限位检查 v1 就是漏了这一步，把"配置限位的数值"直接当电机指令下发，
结果对 `motor_sign=−1` 的关节测了【反方向】（而且没人发现，直到操作员指出
"右边 ID2 电机是大幅度向前、小幅度向后"，与记录矛盾）。
**本轮我在角度换算上错了 3 次**（rpy 约定一次、髋pitch 两次），
**而操作员的物理观察一直是对的。**

### ⚠️ 限位检查的两个已知偏差（不是故障，但上机要知道）
1. **所有关节都到不了配置限位本身**（差 0.08~0.3 rad）⇒ **真实可用行程比配置值小**
2. **髋roll 力矩 6.1~6.7 N·m**，是其它关节（0.55~2.9）的 2~10 倍 ⇒ **该关节受载最大**

---

## 五、✅ 上机路径已定：39 维策略（2026-09-17 晚）

**决策**：上机跑**现成的 39 维 `dm10_joystick_flat`**，**不重训、不写估计器**。
**依据**：65 维缺 `base_lin_vel` 源；39 维观测源齐全。

### 已修 5 项（详见 `robot-deploy-toolkit/results/dm10_39dim_deploy_ready_20260917.json`）

| # | 问题 | 状态 |
|---|---|---|
| **A1** | 🔴 **板子的 `default.yaml` 是改前版本**（obs 顺序修复从未同步）| ✅ 已同步 |
| A2 | 板上 onnx 无 `obs_manifest.json` | ✅ 已生成并上板 |
| A3 | 防呆有效性 | ✅ `check_contract` 双向：新 PASS / **旧 DENY** |
| A4 | `act_alpha` 1.0（不平滑）| ✅ 改 0.8 |
| A5 | 需证明配置真被加载 | ✅ **从运行中的节点读回参数**（不只看磁盘）|

**⚠️ 干跑必须用 launch 文件**：`ros2 launch roboparty_inference inference.launch.py robot:=dm10 policy:=default`
（直接 `ros2 run` 会 fallback 到 `robots/rpo/models/policy.onnx`，报 `780 vs 39` 不匹配）。

### ⚠️⚠️ 上机前必知 3 条

1. **出事【没有中间档】⇒ 机器人会瘫软**。所有保护都是 `FATAL → shutdown() → 失能`
   （`obs_manager.cpp:203-207` 跌倒、`:225-231` 越限）。**吊着没事，落地会摔。**
2. **板上 `policy.onnx` 来源不明**（md5 `359395ad…` 匹配不到本地任何文件，**不是**记录中的"最佳"
   `2026-09-09_00-52-12`）。只影响可追溯性。
3. **7 个安全缺口全未修**（`docs/09-参考/已知限制.md`）。

---

## 五之二、⏸ 65 维路线的状态（本次不走，仅存档）

`base_lin_vel` 消融：置 0 崩 −57%、**滞后一步真值只 −2.5%** ⇒ **A 路（做估计器）可行**。
但"重训 56 维 vs 补估计器"这个决定**还没做**。

详见 memory `dm10-65dim-undeployable-obs-gap`。

---

## 六、产物清单

| 文件 | 内容 |
|---|---|
| `robot-deploy-toolkit/results/motor_sign_20260917.json` | **10 台判定 + 依据 + 置信度** |
| `robot-deploy-toolkit/results/dm10_model_directions.json` | 模型期望方向表（三项自洽校验通过）|
| `robot-deploy-toolkit/scripts/probe_direction.py` | 方向探针（含 BusSniffer 自解反馈）|
| `robot-deploy-toolkit/scripts/export_model_directions.py` | 生成模型期望表 |
| `UniLab/_dir_compare_all.png` | 10 关节方向对照图（①–⑩ 中文标注）|
| `UniLab/_knee_direction.png` | 右膝 直腿 vs 屈膝 对比 |
| `robot-deploy-toolkit/results/limit_check_20260917.json` | **10 台限位/行程检查结果** |
| `robot-deploy-toolkit/results/preflight_singlejoint_20260917.json` | **上机前单关节端到端复核（10/10）** |
| `robot-deploy-toolkit/results/dm10_39dim_deploy_ready_20260917.json` | **39 维策略上机就绪（A1~A5）** |
| `robot-deploy-toolkit/results/can_interface_change_20260917.json` | **CAN 接口改名 + 板载禁用** |
| `robot-deploy-toolkit/results/pd_stand_fallback_20260917.json` | **PD 站立保底（8 项验证）** |
| memory `dm10-frame-alignment` / `dm-driver-error-id-bug` | 结论 + 5 个错误 |

## 七、⭐ 2026-09-17 深夜：PD 站立保底 + **首次真机跑策略**

### 7.1 PD 站立保底（已完成，8 项实测通过）

**把异常处理从 fail-stop（`shutdown` → 瘫软）改成 fail-safe（切 PD 站立）。**

| # | 设计要点 |
|---|---|
| 1 | **新增 `start_mode` 参数，默认 `pd_stand`**（安全优先）。要跑策略需显式切 |
| 2 | ⭐ **检测放在 `control()` 线程（400Hz / 2.5ms）**，不是推理线程（50Hz / 20ms）—— 跌倒时推理可能已发散 |
| 3 | ⭐ **PD 模式绕过 `is_running_`**（独立下发路径）。副作用（刻意）：进 PD 后按 B 暂停不会松掉，要松掉按 X 失能 |
| 4 | ⭐ **不自动切回策略**（决定性实验确认）|
| 5 | **NaN 守卫** `std::isfinite(gravity_b.z())` —— IMU 掉线时四元数全 0 → NaN，而 **NaN > 阈值恒 false** ⇒ 静默失去保护 |
| 6 | `obs_manager.cpp` 两处 `shutdown()` → 切 PD，**且不 throw**（抛异常会杀推理线程）|
| 7 | 两个 service：`/switch_to_pd_stand`、`/switch_to_policy` |

**⚠️ 仍未验证**：**PD 保底【落地实测】** —— 只验了"能触发切换"，**没验"切了之后真能站住"**。
**⚠️ 切换平滑度未调**（`act_alpha=0.8` ⇒ ~12ms 到 90%，可能太快）。

**改动文件**：`inference_node.hpp/.cpp`、`obs_manager.cpp`、`ros_interface.cpp`、`configs/default.yaml`（`act_alpha` 1.0→0.8）
**备份**：均 `<file>.bak-pdstand-20260917-224322`
**记录**：`results/pd_stand_fallback_20260917.json`

### 7.2 ⭐ 首次真机跑策略成功

**操作**：`/init_motors` → `/start_inference` → `/switch_to_policy`（或 `start_mode: policy`）

**观察**：**吊着双腿离地时，策略做出了「有节奏的步态动作」** ✅
（⇒ 策略加载、使能、推理、action 映射 全链路通了）

**⚠️ 但操作员因"怕机械承受不住"断电了**（判断正确 —— 空中行走无地面反力，电机持续满载）。

**⚠️ 仍未知**：
- **策略方向对不对**（腿动得对不对）—— 还没验
- **落地跑**什么表现

### 7.3 ⭐ 顺带发现并工具化：`gs_usb` TX 假死

**断电/再上电后，CAN 接口会进 TX 假死** —— 接口显示 `UP`/`ERROR-ACTIVE`/`bus-off=0`，
**发送不报错但帧不出引脚** ⇒ 扫描**全假阴性**（误判成电机坏）。

**⭐ 两个判据**：① `tx_packets` 不增长 ② **温度读到 0.0°C**（电机不可能 0°C）

**✅ 已做进工具**：`_common.check_tx_alive()` / `check_tx_alive_all()`；
**`sample_motors.py` 每次运行先检测，假死则拒绝继续**（退出码 2）。
**⚠️ 但 `inference_node`（C++）自己不检测** —— 上机时若 canX 假死，节点正常启动但观测恒为 0。

---

## 八、⭐⭐ 2026-09-18 凌晨：首次真机跑策略 —— 三层对照，结论已定

**背景**：上一节（§7.2）只确认了"有节奏的动"，**无法判断对错**。
本轮用**三层对照**把它变成可判定的问题。

### 8.1 ⭐ 先造「标准答案」：仿真参考跑（纯离线）

**在 UniLab 里用【部署的那个 onnx】**（不是 `policy.pt`）跑标称条件：

- **关掉全部 7 个域随机化事件**（真机第一次跑是标称条件）
- **清掉全部 4 处观测噪声**
- **固定命令 `vx=0.4`**（训练分布 0.3~0.5 的中值）
- **从 `home` 关键帧静止起步**（对应真机"PD 站到默认角"）

⚠️ **每个开关都从【活对象】读回验证真的生效**（不是"我以为生效了"）：
```
✅ root 在原点、z=0.693（无位姿随机化）
✅ root 初速为 0 ✅ gravity=[0,0,-1]（无噪声）
✅ dof_pos_rel=0 ⇒ 默认角 = home 位姿  ← 这一条把"推断"变成"实测"
✅ 关节角 = [-0.4,0,0,0.8,-0.4]×2
```

**产物**：`robot-deploy-toolkit/results/dm10_ref_run/`
（`ref_side.mp4` / `ref_iso.mp4` / `ref_trace.npz` / `reference_criteria.json`）

### 8.2 ⭐⭐ 结论一：**策略本身是「蹭地走 + 高频大动作」**

**仿真硬指标**（`reference_criteria.json`）：

| 指标 | 实测 | 目标 | 判定 |
|---|---|---|---|
| 抬脚高度 | **左 1.3 cm / 右 1.2 cm** | 8 cm | ❌ 差 6 倍 |
| **腾空占比** | **0.0% / 0.0%** | >20% | ❌ **一次没离地** |
| 接触滑移 | 0.44 m/s | 越小越好 | ⚠️ ≈ 前进速度 ⇒ **纯打滑** |
| 前进速度 | 0.387 m/s | 0.40 | ✅ 跟得上 |

**⇒ 脚从头到尾没离地，靠在地上滑动前进。** 这就是「蹭地走」。

**相位**：左髋pitch 0° / 右髋pitch 143° ⇒ **左右交替是对的**（不是同步跳）。
**所以不是"没节奏"，是"迈了但没抬脚"。**

**⚠️ 而且策略输出在剧烈抖动**：
```
t=0.30s  左髋pitch act = +2.442
t=0.40s  左髋pitch act = -2.108     ← 0.1 秒内反向
但 q 只在 ±0.1 内动 ⇒ 策略发大指令，电机只能执行不到 1/10
```

**⇒ 「白噪声动作」的老毛病**（memory `unilab-speed-reduction-improves-gait`
提到的 `_diag_contact_jitter.py` 早量到过：`mean|Δaction|=1.30、lag-1 自相关+0.10`）。

### 8.3 ⭐⭐⭐ 结论二：**软件链路是正确的**（这是本轮最硬的成果）

**方法**：从真机 bag 离线**重建 obs** → 喂**同一个 onnx** → 和真机 `/action` 逐点比。

```
帧 0:
  离线 target  [ 0.629  0.828 -0.476 -0.157  0.192 -0.369 -0.151 -0.794  0.668 -0.366]
  真机 /action [ 0.442  0.836 -0.336 -0.016  0.192 -0.200 -0.406 -0.657  0.720 -0.132]
  差           [ 0.187 -0.007 -0.141 -0.141 -0.000 -0.169  0.255 -0.137 -0.052 -0.234]

前 40 帧: 平均|差| 4.8°   最大 23.3°
```

**⇒ 量级、符号、模式全部吻合。**
残差来自 `cmd_vx` 未知（真机用摇杆推的，我按 0.4 假设）+ `last_action` 递推误差。

**⚠️ 这一条排掉了一整类问题**：obs 顺序、`action_scale`、default offset、
onnx 输入输出、`motor_sign`、关节顺序 —— **全部正确**。

### 8.4 ⭐ 关键操作细节：手柄按键表（**实测**，不是推断）

**⚠️ 我按 Xbox 标准布局【推断】键位，连错两次。** 实测结果：

| 手柄键 | 代码索引 | 功能 | 日志字样 |
|---|---|---|---|
| **`X`** | `buttons[2]` | **使能 ↔ 失能** ⭐ | `Motors initialized/deinitialized` |
| **`A`** | `buttons[0]` | **摆到默认位** | `Motors reset` |
| **`B`** | `buttons[1]` | **推理 启动/暂停** | `Inference started/paused` |
| **`Y`** | `buttons[3]` | 手柄 ↔ `/cmd_vel` | `Controlled by joy/cmd_vel` |

**摇杆**：右摇杆(axes[4])=vx，左摇杆(axes[3])=vy，扳机(axes[2]/[5])=转向。
**⚠️ 摇杆满推 = 1.0，训练只见过 0.3~0.5 ⇒ 只推一半。**
**`is_joy_control_` 默认 `true`**（不用按 Y）。

**⚠️ 关键陷阱**：`act_mode`（PD↔策略）**手柄切不了**，只能用 service
`/switch_to_policy` / `/switch_to_pd_stand`。
**而且 `Inference started` ≠ 策略在跑** ——
`act_mode=PD_STAND` 时按 B 只有"变硬"、机器人不动（看起来像"没反应"）。

### 8.5 ⚠️⚠️ 结论三：**这一节的原始结论已被 2026-09-18 修正推翻**（读修正段）

> **⚠️⚠️ 下面「执行器跟不上 20.6°」和「真机幅度 3~8 倍」两个数【都建立在一个不成立的等价上】，
> 已在 §8.5.1 修正。原始数字保留在此仅供追溯。**

**原始记录**：由 `/joint_states` vs `/action` 直接算：

```
整体 平均|误差| 20.6°   最大 159.5°
  左髋pitch  平均 36°  最大 159°   ← 电机完全跟不上
  右膝       平均 44°  最大 115°
  左踝       平均 22°  最大 122°
```

**原始记录**：真机动作幅度 = 仿真的 3~8 倍：

| 关节 | 真机幅度 | 仿真幅度 | 倍数 |
|---|---|---|---|
| 左踝 | 2.71 | 0.35 | **7.8×** |
| 右膝 | 3.62 | 0.60 | **6.1×** |
| 左髋p | 3.94 | 1.17 | 3.4× |

---

### 8.5.1 ⚠️⚠️ 2026-09-18 修正：**上面两个数都不成立，我搞错了**

**代码路径实测**（`inference_node.cpp:295` 与 `ros_interface.cpp:693`）：

```cpp
// 真正发给电机的是【平滑后】的 last_act_
last_act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
robot_->apply_action(last_act_);

// 但 /action 发布的是【未平滑】的 act_
action_msg_.position[i] = act_[i];
```

| | 我比较时用的量 | **真正发给电机的** |
|---|---|---|
| 仿真 | `target = act*0.25 + default` | 同一个（**仿真无平滑**）|
| 真机 | `/action` = `act_`（**未平滑**）| **`last_act_`**（α=0.8 的 EMA）|

**⇒ 我拿「真机未平滑值」比「仿真直接值」—— 不是同一个量。**
**`last_act_` 是 `act_` 的低通 ⇒ 真正施加的幅度比 `/action` 记录的更小**
**⇒ 「3~8 倍」夸大了真机实际动作；「20.6° 误差」也同因被高估**
（高估多少无法确定，因为 `last_act_` 没被记录）。

### 8.5.2 ⚠️ 另一个已撤回的结论：**「IMU 零偏 18°」是错的**

我一度用「最静止的 100 帧」推出 `gravity_b` 有 18° 零偏。**但那段数据里没有静止段**：

```
|ω| < 0.1 rad/s → 只有 1 帧      |ω| < 0.05 → 0 帧
全程 |ω| 中位数 = 3.14 rad/s
```

**⇒ 那 18° 是机器人真的在倾斜/晃，不是零偏。**
**教训：读数的前提假设不成立时，数值毫无意义。**

### 8.5.3 ✅ 已证伪的两个假设（省得再试）

工具：`robot-deploy-toolkit/scripts/sweep_cmd_vx.py`

| 假设 | 实测结果 |
|---|---|
| 真机 `cmd_vx` 比 0.4 大（摇杆推太多）| ❌ **反了** —— 扫 vx 残差最小在 **0.30**（3.63°），随 vx 单调上升（0.5 → 5.54°）|
| 真机 `joint_vel` 噪声大 ⇒ 策略敏感 | ❌ **反了** —— 平滑后残差**变大**（不平滑 3.63° → 平滑 5 帧 4.41°）|

**⇒ 「幅度大」不是观测侧造成的。**

### 8.5.4 ⭐ 由此浮现的**真实 sim2real 失配**（新的待办）

**训练：目标直接生效。部署：目标经过 α=0.8 的一阶低通。**
⇒ **真机闭环回路上多了一个仿真里不存在的低通环节。**

**⚠️ 立项时（§5 A4）把 `act_alpha` 1.0→0.8 的理由是「文档建议防动作突变冲击」，
但没意识到它改变了闭环动力学。**

**两个待办**：
1. `act_alpha` 要么改回 `1.0`（消除失配），要么**训练侧也加同样平滑**
2. **`/action` 应发布 `last_act_`**（或两者都发），否则记录的不是真正下发的东西

### 8.5.5 ⚠️ 一个真实但未解释的数

```
全程 |linacc| 均值 = 20.78 m/s²   ← 重力 9.81 的 2.1 倍
```

**⇒ 机器人在剧烈冲击/振荡**（与「act 高频抖动 + 执行器跟不上」的描述一致）。

### 8.6 ⭐ 顺带钉死：`policy.onnx` 出处

**方法**：权重指纹（不是字节哈希 —— onnx 导出会因环境产生不同字节）。

```
部署的 onnx 权重指纹 ≡
  UniLab/logs/rsl_rl_ppo/DM10JoystickFlat/2026-09-09_00-52-12_mujoco/policy.onnx
```

**而且它正好是「去掉绊脚惩罚、恢复迈步」那版**：

| run | `penalty_close_feet_xy` | 备注 |
|---|---|---|
| `2026-09-09_00-19-23` | −0.2 | 备份名里的 `prev_pen02` |
| **`2026-09-09_00-52-12`** ← 部署 | **（无）** | 训练配置注释里的 "00-52-12 baseline" |

**ONNX 结构**：`39 → 512 → 256 → 128 → 10`，ELU；
**`empirical_normalization` 已烤进图里**（`Sub` → `Div`：`(obs−mean)/std`）
⇒ **部署侧不需要再做归一化**（这是很可能静默出错的地方，已排除）。

### 8.7 ⚠️ 本轮踩的坑（我的错误）

1. **手写 CDR 反序列化错了 5 次** —— `sensor_msgs/JointState` 的变长字符串
   对齐（`action_N` 8/9 字节 vs `joint_N` 7/8 字节 ⇒ 后续 offset 落点不同），
   纯推 offset 极易错。**最后用板子上 `rclpy` 官方反序列化一次就通。**
   ⇒ **教训：有官方库时别手写序列化格式。**
   工具：`robot-deploy-toolkit/scripts/export_bag_on_board.py`（在板子上跑）

2. **按键映射推断两次** —— 见 §8.4。

3. **onnx 权重指纹跨脚本不可比** —— 我两个脚本用了不同序列化（`arr.shape`
   元组 vs `list(shape)`），哈希当然不同。**指纹只在同一函数内可比。**

### 8.8 产物清单（本轮新增）

| 文件 | 内容 |
|---|---|
| `robot-deploy-toolkit/results/dm10_ref_run/ref_trace.npz` | **仿真参考轨迹（q/act/target/obs/足端/底盘）** |
| `robot-deploy-toolkit/results/dm10_ref_run/ref_side.mp4` / `ref_iso.mp4` | 侧视 / 斜视视频 |
| `robot-deploy-toolkit/results/dm10_ref_run/reference_criteria.json` | **机读判据（摆幅/频率/相位/抬脚）** |
| `robot-deploy-toolkit/results/dm10_real_run/real_run.db3` | **真机 bag（原始）** |
| `robot-deploy-toolkit/results/dm10_real_run/real_run_export.npz` | **真机数据（官方反序列化）** |
| `UniLab/_make_reference_run.py` | 仿真参考跑（含"从活对象读回"验证） |
| `robot-deploy-toolkit/scripts/export_bag_on_board.py` | ⭐ **bag→npz（板子上跑，官方 rclpy）** |
| `robot-deploy-toolkit/scripts/analyze_real_vs_onnx.py` | ⭐ **决定性对照脚本** |
| `robot-deploy-toolkit/scripts/_trace_policy_provenance.py` | onnx 出处追踪（权重指纹） |

---

### 8.9 ⭐⭐⭐ 根因实测确认：**`feet_air_time` 的命令门控【恒 False】**

**症状**：真机与仿真**都**是抬脚 1.3cm、腾空 **0.0%**、滑移 0.44 m/s ≈ 前进速度。
**⚠️ 而软件链路已证明是对的（4.8°）** ⇒ **瓶颈是策略本身不会抬脚。**

**根因**（`src/unilab/tasks/locomotion/common/gait_terms.py:60-71`）：

```python
def _command_gate(env, term, command_name, command_threshold):
    total = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
    return np.asarray(total > command_threshold)      # ⚠️ 严格大于
```

**配置**：`feet_air_time.command_threshold: 0.5`
**命令采样范围**：`lin_vel_x: [0.3, 0.5]`（上界恰好是 0.5）
⇒ **门控只在 `vx > 0.5` 时为真，采样取不到 ⇒ 恒 False**
⇒ **`feet_air_time` 权重（0.25 / 0.8）恒 ×0**

**⭐ 直接调生产代码 `_command_gate` 的实测**：

```
command_threshold = 0.5    ⇒ 训练分布内 0/6 为真   ← 死的
command_threshold = 0.05   ⇒ 训练分布内 6/6 为真   ← 修复有效
```

**⭐ 独立铁证**：tensorboard 里 **4 条 run 的 `reward/feet_air_time` 全部 = `0.000`**，
而**同为命令门控**的 `feet_clearance`(thr=0.01) / `feet_slip`(thr=0.01) /
`feet_contact_number`(thr=0.01) **都有正常非零值**。
⇒ **只有 `feet_air_time` 被"边界差一个等号"杀死。**

**修法**：`command_threshold: 0.5 → 0.05`（与同族项一致）。
**⚠️ 重训前必须做短程 sanity（~100 iter）验证 `reward/feet_air_time ≠ 0`。**

#### ⭐ 同类陷阱（值得单独记）

**边界差一个等号 ⇒ 条件永不满足，但训练正常跑完、reward 正常上升、不报任何错。**
**只有去读 tensorboard 里该项的数值才能发现。**

**⇒ 检查任何"带阈值的门控/裁剪"时必须问**：
1. 阈值和变量的取值范围**有重叠吗**？
2. 是**严格不等**还是**非严格**？**边界值取得到吗**？

#### ✅ 附带结论：**换 run 无用**（已证明，别再试）

26 个 run 里**只有 4 个与部署配置完全兼容**，且它们与部署的那条**统计上无法区分**：

| run | `feet_clearance` | `feet_phase_contrast` | `tracking_lin_vel` | `feet_air_time` |
|---|---|---|---|---|
| `00-52-12`（**部署的**）| −0.056 | 1.492 | 1.926 | **0.000** |
| `14-41-30` | −0.059 | 1.493 | 1.944 | **0.000** |
| `15-35-14` | −0.051 | 1.491 | 1.905 | **0.000** |
| `15-55-50` | −0.043 | 1.489 | 1.934 | **0.000** |

**⇒ 换成任何一条都会同样蹭地走。**
**⚠️ 排除 `13-59-20`**：用 `ctrl_dt=0.01`（100 Hz），与部署的 `decimation=5 × dt=0.004 = 50 Hz` **不兼容**。

**⇒ 「换策略」这条路已关闭，只能重训。**

---

## 九、下一步（按优先级）

| # | 做什么 | 为什么 | 成本 |
|---|---|---|---|
| **1** | ⭐ **改 `feet_air_time.command_threshold: 0.5 → 0.05`** + **短程 sanity（~100 iter，验 `reward/feet_air_time ≠ 0`）** | §8.9 的根因修复；**不验 sanity 就是又一次静默不生效** | 30 min |
| **2** | **全量重训** | 拿到真会抬脚的策略 | ~2 h |
| **3** | **用 `_make_reference_run.py` 出参考跑 + 判据**（抬脚 >5cm、腾空 >20%）**并看视频** | ⚠️「指标达标 ≠ 观感好」 | 20 min |
| **4** | **PD 落地站立测试** | 零风险，且是安全退路（§7.1 未验项） | 30 min |
| **5** | **`act_alpha` 决策**（改回 1.0 或训练侧加同平滑）+ **`/action` 改发 `last_act_`** | §8.5.4 的 sim2real 失配 | 需先给方案 |
| 6 | 65 维路线 | 需 `base_lin_vel` 估计器（§5）；**但当前 39 维本身就不会走路，先解决这个** | — |

**⚠️ 已关闭的方向（别再试）**：
- **换 run**（§8.9，4 条兼容 run 统计上无区别）
- **改部署侧代码**（软件链路已证明正确，4.8°）
- **cmd_vx / joint_vel 噪声**两个假设（§8.5.3 已证伪）
- **手写 bag 反序列化**（已错 5 次；必须用 `export_bag_on_board.py`）

**⚠️ 上机前必读**（§7.2 原有）：
- 出事无中间档（PD 保底已做，但"落地站住"未验）
- **手柄 `X` = 一键失能**（§8.4）
- **`act_mode` 手柄切不了，必须用 service**
- 干跑必须用 **launch 文件**（`ros2 run` 会 fallback 到 rpo 报 780vs39）

## 十、⭐⭐ 2026-09-18 续：根因修复 + 重训验证 —— **策略真的会走了**

### 10.1 ⭐ 修复（一个数字）

**文件**：`src/unilab/conf/ppo/task/dm10_joystick_flat/base.yaml`
```yaml
  feet_air_time:
    params:
      command_threshold: 0.5    →    0.05     # ⭐ 唯一改动
```
**依据**：`_command_gate`（`common/gait_terms.py:60-71`）判据是**严格大于**
`total > command_threshold`，而 `total = |v_xy| + |w_z|`，
训练命令采样 `lin_vel_x=[0.3,0.5]` ⇒ 上界恰好 0.5 ⇒ **恒 False**。
**改 0.05 后实测 0/6 → 6/6**（直接调生产代码验证）。

**⚠️ 短程 sanity 是硬门槛**（否则又是"静默不生效"）：
跑 50 轮后读 tensorboard ⇒ `reward/feet_air_time` **非零 50/50** ✅

### 10.2 ⭐⭐ 重训结果：**质变**

**run**：`logs/rsl_rl_ppo/DM10JoystickFlat/2026-09-18_01-24-25_mujoco`
（3000 轮，seed=1；`policy.pt` + `policy.onnx` + `play_video.mp4` 均已产出）

**奖励项走势（全程 3000 轮）**：

| 项 | 早期 | 最终 | 旧策略（改前）|
|---|---|---|---|
| **`feet_air_time`** | +0.0393 | **+0.1498** | **0.0000** ⭐ |
| `tracking_lin_vel` | +1.4640 | +1.9264 | 1.9341（未牺牲）|
| `feet_phase_contrast` | +1.1649 | +1.4926 | 1.4886（持平）|
| `feet_clearance` | −0.0268 | −0.0293 | −0.0434 |

**步态量化**（`_measure_gait.py`，1000 步 = 20 s）：

| 指标 | 旧策略 | **新策略** | 目标 | |
|---|---|---|---|---|
| **腾空占比** | **0.0%** | **50.4% / 49.8%** | >20% | ✅✅ |
| **抬脚高度** | 1.3 cm | **19.8 cm(左) / 8.2 cm(右)** | >5 cm | ✅ |
| 接触滑移 | 0.44 m/s | **0.154(左) / 0.387(右)** | <0.15 | ⚠️ 右未达标 |
| 前进速度 | 0.387 | **0.557 m/s** | ≈0.41 | ⚠️ 超 36% |
| 步频 | — | **3.85 Hz** | ~1.5 Hz | ⚠️ 偏快 2.5× |
| base 高度 | — | 0.695 ± 0.022 | — | ✅ 稳 |

**⭐ 视频确认**（`sheet_new_policy.png`，`play_video.mp4`）：
**清晰的交替步态** —— 一腿支撑、另一腿抬起前摆，躯干竖直不倒地。
**与旧策略「腿拖在地上蹭」是质的不同。**

### 10.3 ⚠️ 本轮踩的坑（我的错误）

| # | 错误 | 教训 |
|---|---|---|
| **1** | ~~**三条 seed 并行训练 ⇒ CUDA `unspecified launch failure`（@2181 轮）**~~ **⚠️ 2026-09-19 推翻** | **真因是系统 s2idle 休眠**，不是 CUDA 崩，也跟并行无关。journal 原文：`01:07:21 sleep requested` → `01:07:28 Entering sleep state 'suspend'` → 三条进程同时被杀；唤醒时 NVIDIA 报的是 **Xid 31 MMU Fault**（来自 resume），**根本没有 `unspecified launch failure`**。**⇒「改串行」是无效对策** —— 该做的是训练期间禁止休眠（已加进 `train_safe.sh` 的 `systemd-inhibit`）。**并行从未被证明有问题，是个还没试过的方向** |
| **6** | ⭐⭐ **01:24 启动的那条 run 静默跑在 CPU 上** | 休眠唤醒后 `torch.cuda.is_available()=False`，而 `training.device` 没写死 ⇒ `get_default_device()` **静默返回 `cpu`**，零报错。整场 3000 轮慢 **2.3 倍**（55 分钟 vs 24 分钟）—— 全历史 60+ 条 run 里唯一一条跑 CPU 的，正是唤醒后 4 分钟启动的那条。**⇒ 现在一律走 `train_safe.sh`**（启动前校验 CUDA + 禁止休眠）；想彻底关掉静默降级就在 owner yaml 里写死 `training.device: cuda` |
| **2** | ⭐ **用进程数判断"在跑"** —— 三条进程在但**早崩了**，我还报"ETA 13 分钟" | **用户一句「cpu没有负载」点破**。正解=**看 GPU 利用率/产物时间戳**，不是 `pgrep`。这是 memory `silent-nonbinding-conditions` 的老毛病重犯 |
| **3** | **导出 `policy.pt` 试了 4 次全失败** | ①`cli eval --load-run` flag 不通 ②`algo.load_run=` Hydra 解析错 ③`render_run.py` 是 off-policy 路径 ④手工重建 runner 缺 `normalize_ppo_train_cfg`/`RslRlVecEnvWrapper`/`apply_ppo_runtime_flags` 一整条栈。**⇒ 别手工重建框架初始化栈；训练正常跑完会自动导出，别为省 30 分钟折腾 4 轮** |
| **4** | **读脚本自报的 ETA 当准**（说 13 分钟，实际 53 分钟） | 要**交叉验证速率**（轮数/时间） |
| **5** | `_export_ckpt.py` 里假设 `run_config.json` 顶层是配置 | 实际是 `{run, config, contract_snapshot}`，配置在 `["config"]` |

### 10.4 ⚠️ 三个待解（**只有 1 seed × 1 episode，不能下最终结论**）

1. **左右严重不对称**：抬脚 **19.8 vs 8.2 cm**（差 2.4 倍）、滑移 0.154 vs 0.387
   ⇒ 可能真学了"瘸腿走"，也可能是采样噪声。**需多 seed / 多 episode 确认**
2. **步频 3.85 Hz**（训练奖励的 `feet_phase` 频率是 **1.5 Hz**）⇒ 快 2.5 倍，疑似小碎步
3. **速度超指令 36%**（0.557 vs 0.41）

### 10.5 ⚠️ 物理参数从未更新

`reports/DM10下肢_实机参数登记表.md`：12 个 body 的**质量实测列全是空的**。

| 项 | 仿真值 | 实测 |
|---|---|---|
| 整机总质量 | **10863.1 g** | ⬜ 用户口述约 **5200 g**（**差一倍**）|
| 惯量 | 从形状估算 | ⬜ |
| 几何尺寸（8 项）| 从 XML 换算 | ⬜ |

**⚠️ 差一倍是很大的 sim2real gap。但注意：它不是「蹭地走」的原因**
（那是奖励项坏了），**是"走起来之后稳不稳"的问题。**

**⇒ 下一步值得做，但别和奖励修复混为一谈。**

### 10.6 产物清单（本轮）

| 文件 | 内容 |
|---|---|
| `logs/rsl_rl_ppo/DM10JoystickFlat/2026-09-18_01-24-25_mujoco/` | ⭐ **新策略**（policy.pt/onnx + play_video.mp4）|
| `robot-deploy-toolkit/results/dm10_ref_run/sheet_new_policy.png` | 新策略视频抽帧 |
| `robot-deploy-toolkit/results/dm10_real_run/` | 真机数据（不变）|
| `UniLab/_export_ckpt.py` | 导出尝试（4 版，**未成功**，留作参考）|
| `robot-deploy-toolkit/scripts/sweep_cmd_vx.py` | cmd_vx 假设检验（证伪用）|

---

## 十一、下一步（按优先级）

| # | 做什么 | 为什么 |
|---|---|---|
| **1** | **补跑 2~3 条 seed**，确认「不对称 / 高步频」是否稳定 | ⚠️ **单 seed 不能下结论**（我犯过） |
| **2** | **新策略上机试跑** | 现在是**真的会走**（会真的移动位置）⇒ **安全要求更高**：吊着或留行程 + 手不离 `X` |
| **3** | **物理参数实测**（称重 12 个 body + 量尺寸）| sim2real gap 差一倍 |
| **4** | 若不对称/步频是稳定的 ⇒ 调奖励（加强左右对称、降步频）再重训 | |
| **5** | `act_alpha` 决策（§8.5.4 sim2real 失配）+ `/action` 改发 `last_act_` | |

**⚠️ 已关闭的方向**：换 run（§8.9）/ 改部署侧代码（软件链路已证明对）/
cmd_vx 与 joint_vel 噪声假设（§8.5.3 已证伪）/ 手写 bag 反序列化（必用 `export_bag_on_board.py`）

**⚠️ 上机安全（实测键位）**：
```
X = 一键失能（保命）    A = 摆默认位    B = 推理启停    Y = 切 joy/cmd_vel
act_mode 手柄切不了 ⇒ 必须 ros2 service call /switch_to_policy
Inference started ≠ 策略在跑（PD 模式下按 B 只"变硬"）
摇杆满推=1.0，训练只见过 0.3~0.5 ⇒ 只推一半
```
