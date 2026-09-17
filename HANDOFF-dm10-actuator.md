# 交接：dm10 执行器建模（新会话从这里开始）

> **创建于** 2026-09-17。上一会话上下文约 520K/1M，在此交接。

---

## 一、当前 git 状态（已就绪，无需再动）

```
分支:   dm10          （基于 upstream/main = 84a00d04）
提交:   67e1e141      dm10: 双足人形任务（配置 + 奖励项 + 资产 + 测量脚本）
远端:   origin   → git@github.com:Tensor-0/UniLab.git      你的 fork
        upstream → git@github.com:unilabsim/UniLab.git     上游（会重定向到 Motphys）
快照:   /home/zhan/unilab-dm10-snapshot-20260917-1141.tar.gz  (4.5 MB)
```

**工作区干净**（`git status` 无未提交改动）。
**三层保护**：快照 + 本地分支 + fork 远端。

⚠️ **动手前不要再动 git**。有安全网了，直接改代码即可。

---

## 二、要做什么

**Q2：给 dm10 接上真实的执行器模型（τ-ω），让仿真更接近真机。**

背景：现在 `dm10.xml` 里是 `${10 个 <position> kp=30 kv=3.0, forcerange=±28}$`，
这是个**与速度无关的常数力矩上限** —— 而真机的力矩随转速明显下降。

---

## 三、已完成的准备工作（Q1，全部有实测证据）

### 3.1 达妙 DM-J4340-2EC (24V) 的真实参数

**仓库**：`~/dm-actuator-model/`（GitHub: `Tensor-0/dm-actuator-model`）

| 项 | 值 | 来源 |
|---|---|---|
| **τ-ω 直线** | `ω = −0.118555·τ + 6.401745` rad/s | 官方台架曲线 97 点拟合，**r²=0.9916** |
| 空载转速 | 6.4017 rad/s (61.1 rpm) | 实测外推 |
| **堵转扭矩** | **54 N·m** | ⚠️ **实测选定**，非说明书标称的 40 |
| 连续扭矩 | 12 N·m | 额定 |
| MuJoCo 参数 | `nominal="24 54 6.4017"` | 三个数即可 |

**⚠️ 关键决策记录**：`stall_torque` 用 **54** 不用 40，理由是与官方曲线逐点比对：

| 取值 | 平均误差 | 最大误差 |
|---|---|---|
| **54** | **0.85%** | 3.28% ✅ |
| 40（说明书标称） | 9.94% | 17.00% ❌ |

证据：`~/dm-actuator-model/data/tau_omega_overlay.png`（叠加图）

### 3.2 MuJoCo 原生 `<dcmotor>`（⭐ 已实测可用）

```xml
<dcmotor name="leg_l1_joint_motor" joint="leg_l1_joint"
         nominal="24 54 6.4017"
         input="position"              <!-- ⚠️ MuJoCo 3.11 写法 -->
         controller="30 0 3 0 0 24"    <!-- kp ki kd slewmax Imax Vmax -->
         forcerange="-28 28"/>
```

**已经实测验证**（MuJoCo 3.11.0，UniLab 的 `.venv`）：
- 空载稳态 `ω = V/K`，扫 24/18/12/6/3 V **误差 0.00%**
- `position` 模式跟踪 0.5 rad → 收敛到 **0.4999**
- **τ-ω 生效**：扫 `Vmax` → 峰值力矩线性变化
- 力矩公式数值反推 `τ = (K/R)v − (K²/R)ω`

**验证脚本**：`~/dm-actuator-model/tools/verify_mujoco_dcmotor.py`、
`verify_dcmotor_loaded.py`、`plot_tau_omega.py`

### ⚠️ 3.3 版本陷阱（务必记住）

| MuJoCo 版本 | `input` 写法 |
|---|---|
| **3.11（本机）** | **`input="position"`** / `"velocity"` / `"voltage"` |
| 3.12+ | `input="pos vel"` 等空格分隔签名 |

**本机 3.11 实测 `input="pos vel ff"` → `XML Error: invalid keyword`**。
别照抄网上的 3.12 示例。

---

## 四、⭐ 上游已有完整先例（照抄架构）

**`~/dm-actuator-model/reference/bam_action.py`**（451 行）
来源：`rocPAI-Forge/microduck_rl_unilab`，源自 `Motphys/UniLab` PR #1475

**细读文档**：`~/dm-actuator-model/docs/bam_action_源码细读_20260917.md`

### 核心架构（已对本机框架代码核实）

```python
class XxxAction(ActionTerm):
    requires_substep_state_feedback: ClassVar[bool] = True    # ← 关键声明

    def apply_actions(self) -> None:
        # 每个【物理 substep】被调用
        ...
        self._entity.data.write_ctrl(torque, actuator_ids=self._actuator_ids)
```

框架侧 `src/unilab/envs/manager_based_rl_env.py:385-410` 自动接线：
```python
if not self.action_manager.requires_substep_state_feedback: return
self._backend.set_pre_step_control(self._apply_manager_control)
```
**⇒ 零框架契约变更。**

### 为什么必须逐 substep
dm10：`sim_dt=0.005`（200Hz 物理）/ `ctrl_dt=0.02`（50Hz 策略）。
τ-ω 限幅依赖**瞬时 q̇**；只在 50Hz 算会让中间 4 个 substep 共用常值 → 高速段严重失真。

---

## 五、⚠️ 已知的框架限制（会直接影响方案）

### 5.1 UniLab 读不到真力矩

实测确认 `unisim` backend **没有任何** `actuator_force` / `qfrc_*` getter；
`env._backend` 里找不到 `MjData`；`set_pre_step_control` 钩子只给 `(backend, ctrl)`。

**⇒ 力矩只能"重构"**（`τ = kp(ctrl−q) − kd·q̇` 再截断），这也是上游的做法。

**⚠️ 更关键**：换成 `<dcmotor>` 后 `gainprm/biasprm` 语义变了
（dcmotor 不符合 affine 结构），**该重构公式只在 baseline 上成立**。

**解决方案（学上游）**：让 action term 自己把算出的力矩**存下来并暴露**
（`bam_action.py` 的 `applied_torque` property，注释写着
"Closest observable to the upstream actuator_force"）。

### 5.2 sim2sim 契约拦不住执行器改动

`src/unilab/utils/sim2sim.py`：
- `ALLOWLIST` 含 `env.scene`（actuator 就在 scene XML 里）
- `DENYLIST` **无任何 actuator 字段**

**⇒ 换执行器不报错、不警告、静默通过。** 要自建校验。

---

## 六、评估方法（Q1/Q2/Q3）

**文档**：`~/dm-actuator-model/docs/评估方法_Q1Q2Q3_20260917.md`

| | 问题 | 状态 |
|---|---|---|
| **Q1** | 模型像不像真机？ | ✅ **已完成**（平均误差 0.85%）|
| **Q2** | 换模型后策略还能不能走？ | ⬜ **本次要做的** |
| Q3 | 重训后是否更好？ | ⬜ 以后 |

### ⚠️ Q2 的预期（必须提前知道）

你的历史实测（`~/reports/DM10执行器建模_诊断与方案_20260912.md`）：
**`|q̇|` 峰值 8.0~12.06 rad/s，6/6 片段全部超过 5.0 rad/s**，
而电机空载上限只有 6.4 rad/s。

**⇒ Q2 很可能得出"策略在新模型下会垮"。**
**但那是模型变准的表现，不是策略变差** —— 说明旧的 `forcerange=±28`
一直在掩盖真机做不到的行为。

---

## 七、下一步的建议顺序

1. **先读 `dm10.xml` 当前状态**（`~/UniLab/src/unilab/assets/robots/dm10/dm10.xml:193-210`）
2. **决定路线**（见下）
3. **写 action term**（复刻 bam_action 架构）
4. **跑 Q2 测量**（对比 baseline vs dcmotor）

### 两条路线（★ 需用户拍板）

| | **路线 A：原生 `<dcmotor>`** | **路线 B：`<motor>` + Python 算 τ** |
|---|---|---|
| XML 改动 | 换标签 | 改 `<motor>` |
| τ-ω | MuJoCo 内部算 | 自己算 |
| 反电动势/热/LuGre | **原生支持** | 要自己写 |
| 与现有 kp/kv 冲突 | 有（语义变了） | 无 |
| 代码量 | 少 | ~450 行 |

**A 更省事**（MuJoCo 原生能力更强），但要注意 `gainprm/biasprm` 语义变化对
其他依赖它的代码（如 `joint_torque_l2`）的影响。

---

## 八、必须遵守的纪律（血泪教训）

**memory `unilab-measurement-pitfalls`（七个坑）**：
- seed=0 **且** 清空 DR（`override['events']={}`）
- **必须 32 env**（单 env 的 episode 长度只有 1/3.8）
- done 步**整帧丢弃**（reset 传送污染，滑移虚高 5.5 倍）
- 用**固定步数预算**，不用"凑够 N 个 episode"（右删失）
- **≥4 seed** 才下"显著"结论
- 报数字注明**单腿/两腿**、**池化/逐 episode 等权**

**主判据用 `Episode_Reward/*` 单步归一化值，不用 `mean_reward`。**

---

## 九、相关资产位置速查

| 用途 | 路径 |
|---|---|
| 达妙参数 + 验证工具 | `~/dm-actuator-model/` |
| 上游 action term 参考 | `~/dm-actuator-model/reference/bam_action.py` |
| 架构细读 | `~/dm-actuator-model/docs/bam_action_源码细读_20260917.md` |
| 评估方法 | `~/dm-actuator-model/docs/评估方法_Q1Q2Q3_20260917.md` |
| dm10 模型 | `~/UniLab/src/unilab/assets/robots/dm10/dm10.xml` |
| 现有测量脚本 | `~/UniLab/_measure_*.py`（16 个，**都不读力矩**）|
| 旧执行器报告 | `~/reports/DM10执行器建模_诊断与方案_20260912.md` |

### 相关 memory
- `mujoco-native-dcmotor-exists` —— MuJoCo 原生 dcmotor 能力 + 版本坑
- `unilab-actuator-model-port-precedent` —— 上游先例 + 框架限制
- `unilab-measurement-pitfalls` —— 七个测量坑
- `damiao-feedback-frame-facts` —— 达妙电机实测事实
- `pd-semantics-sim-vs-mit-differ` —— PD 语义差异

---

## 十、开新会话时的第一句话建议

> 读 `~/UniLab/HANDOFF-dm10-actuator.md`，我们继续做 Q2：
> 给 dm10 接上真实执行器模型（τ-ω），先给方案再动手。
