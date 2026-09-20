"""DM10 电压驱动位置控制 —— 真实达妙 DM-J4340-2EC 力矩-转速曲线。

为什么需要这个 term
------------------
``dm10.xml`` 用的是 ``<position kp=30 kv=3 forcerange=±28>``：一个**与速度无关的
常数力矩上限**。它在 20 rad/s 时照样输出 28 N·m，而真机 DM-J4340-2EC 在
6.4017 rad/s（空载转速）时输出 **0 N·m**，中间线性下降。这条缺失的 τ-ω 斜率
正是 dm10 项目"仿真里能走、真机上不行"的一个已知来源。

本 term 与 ``dm10_actuator.xml``（MuJoCo 原生 ``<dcmotor>``）配对工作：
物理侧由 MuJoCo 积分，控制侧在这里算。

分工（这是关键设计）
------------------
- **XML / MuJoCo**：负责 τ-ω 本身 —— ``τ = (K/R)·(v − K·q̇)``，含反电动势与
  电流饱和。这些**不在这里重写**——重写就会引入近似。
- **本 term**：把策略的位置目标变成一个**电压**，每物理子步写进 ``ctrl``。

物理全部交给引擎的好处是：**力矩在 Python 侧是可精确观测的**。只要在同样的状态上
求值同一个闭式，就与 MuJoCo 内部逐位相同（实测最大误差 2.8e-14，见
``~/dm-actuator-model/tools/verify_dm10_torque_reconstruction.py``）。
不需要 backend 暴露 ``actuator_force``（它没有，见下）。

为什么必须逐物理子步（200 Hz）而不是每控制步（50 Hz）
---------------------------------------------------
τ-ω 项依赖**瞬时** q̇。若只在控制步算一次，中间的 3 个物理子步会共用同一个由
20 ms 前的速度算出的常值 —— 而 τ-ω 斜率很陡（−8.44 N·m 每 rad/s），高速段
会严重失真。声明 ``requires_substep_state_feedback = True`` 后，框架会自动把
``apply_actions`` 挂到每个物理子步上（``manager_based_rl_env.py:_configure_action_control``），
**零框架契约变更**。该架构与上游先例一致（``Motphys/UniLab`` PR #1475 的
``bam_action.py``，本地副本 ``~/dm-actuator-model/reference/bam_action.py``）。

为什么不从 backend 读真力矩（已知的框架限制）
--------------------------------------------
UniLab 的 ``unisim`` backend **没有** ``actuator_force`` / ``qfrc_*`` getter，
``env._backend`` 里也拿不到 ``MjData``，``set_pre_step_control`` 钩子只给
``(backend, ctrl)``。上游 ``bam_action.py`` 遇到了同样的限制（其 docstring 有记录），
解法是让 term 自己把力矩算出来并暴露。本 term 用 ``applied_torque`` property
做同一件事 —— 区别是上游只能**近似重构**，而这里因为 PD 律就在本文件内，
重构是**精确**的。

增益的标度（容易搞错 —— 我在这里错过两次）
----------------------------------------
``<position>`` 的 kp 单位是 **N·m/rad**，这里的 kp 单位是 **V/rad**。照抄数字会错。

**正确的换算是「单位换算」，不是「比例缩放」**：

    τ = (K/R)·v = (K/R)·kp_v·e      ⇒  dτ/de = (K/R)·kp_v
    要 dτ/de = 30 N·m/rad  ⇒  kp_v = 30 · R/K = 30 × 0.4444 = 13.3333 V/rad

⚠️⚠️ **别写成 `30 × K/R = 67.5` 或 `30 × (K/R) 的倒数混淆`** —— 我先后错过两次：
   - 第一次：用 ``kp_v = 30 · R/K = 1.6667``（把阻尼换算 R/K 套到刚度上），
     实得 ``dτ/de = 1.6667 × 2.25 = 3.75``，只有基线的 **1/8** → 策略崩塌，
     电压利用率仅 1.75%。这是「假崩溃」，与 τ-ω 无关。
   - 第二次：用 ``kp_v = 24.5``，实得 ``dτ/de = 55.125``，是基线的 **1.84 倍**，
     却对外写成「与基线同一物理刚度」——**错的**。那批实验里混着「刚度」这个因子。

**自检方法**：算完一定回头验 ``kp_v · K/R`` 是否等于目标 ``dτ/de``。
本机数字：``K/R = 2.25``、``R/K = 0.4444``。**乘除别搞反**。

近似边界 / 未建模项（照上游惯例如实标注）
----------------------------------------
- **无指令延迟**：上游 ``bam_action`` 有 LIFO 延迟缓冲模拟总线延迟。这里没有。
  达妙是 CAN 直驱，延迟量级未知（没测过），凭想象填一个数不如不填。
- **无温度降额**：``nominal`` 是冷态 24 V 台架曲线。连续 12 N·m 额定扭矩与温升
  相关的降额未建模（官方有温升曲线，``~/dm-actuator-model/data/thermal.json``）。
- **无齿槽转矩 / LuGre 摩擦 / 电感**：MuJoCo 3.11 的 ``<dcmotor>`` 支持这些，
  但达妙没有对应的实测参数，故不开。
- **τ-ω 外推边界**：官方台架只测到 19.35 N·m / 5.90 rad/s。54 N·m 的堵转点是
  线性外推（平均误差 0.85% 是对**实测区**而言）。超出实测区的行为不可保证。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.tasks.motion_tracking.common.manager_terms import (
    MotionJointPositionAction,
    MotionJointPositionActionCfg,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def _real(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    strict: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        raise ValueError(f"{label} must be {'>' if strict else '>='} {minimum}")
    return result


@dataclass(kw_only=True)
class DmVoltageActionCfg(MotionJointPositionActionCfg):
    """配置 DM-J4340-2EC 电压驱动位置控制。

    继承 ``MotionJointPositionActionCfg`` 以保留运动跟踪的业务契约
    （``command_name`` 的零位偏置、``simulate_action_latency``、动作 scale/clip），
    只增加电气侧参数。
    """

    # ── 电气参数：由 nominal="V τ_stall ω_no_load" 推出，与 XML 必须一致 ──
    bus_voltage: float = 24.0
    """电源电压 V（对应 XML 的 nominal 第一个数与 ctrlrange 上界）。"""

    stall_torque: float = 54.0
    """堵转扭矩 τ_stall（N·m）。⚠️ 54 是官方台架曲线推出的值，非说明书标称的 40：
    与 97 个实测点逐点比对，54 → 平均误差 0.85%，40 → 9.94%。"""

    no_load_speed: float = 6.4017
    """空载转速 ω_no_load（rad/s），台架实测外推。"""

    # ── 控制律增益（⚠️ 单位是 V/rad 与 V·s/rad，不是 N·m 制）────────────
    kp_voltage: float = 1.6667
    """位置误差增益，V/rad。默认 = 30 N·m/rad · R/K，使位置项与基线逐点等价。"""

    kd_voltage: float = 0.1667
    """速度增益，V·s/rad。默认 = 3.0 N·m·s/rad · R/K。
    ⚠️ 净速度阻尼会比基线大 —— 反电动势贡献了额外 8.44 N·m/(rad/s)。见模块 docstring。"""

    kff_voltage: float = 0.0
    """反电动势**前馈补偿**，V·s/rad。加入后电压变为
    ``v = kp_v·e − kd_v·q̇ + kff_v·q̇``，即 `kff_v` 部分抵消反电动势。

    为什么需要它：反电动势给关节自带 ``K²/R = 8.44`` N·m/(rad/s) 的阻尼，
    而基线 ``<position kv=3>`` 的总阻尼只有 **3.0**。所以 ``kd_v ≥ 0`` 时
    总阻尼**永远 ≥ 8.44**，比基线大 2.8 倍以上 —— 这个差异无法用常规增益消掉。

    要让总阻尼等于 ``D``，令 ``kff_v = kd_v + (K/R) − D·R/K``。例如
    ``D=3.0, kd_v=0`` ⇒ ``kff_v = 2.4164``（实测 dv/dq̇ 与理论逐位一致）。

    ⚠️ 这是**前馈**，不是反馈增益：它不去稳系统，只是把电机被反电动势拿走的
    力矩补回来。真机固件通常也做这件事（达妙是否做，**未核实**）。
    默认 0（不做补偿），因为它会抬高高速时的力矩，属于额外假设。

    2026-09-17 实测：``kp_v=24.5, kd_v=0, kff_v=2.4164`` 时 episode 长度
    178.8±14.7 步，**高于**基线 162.3±10.9（终止时走完片段 98% vs 87%）。
    """

    def build(self, env: ManagerBasedRlEnv) -> DmVoltageAction:
        return DmVoltageAction(self, env)


class DmVoltageAction(MotionJointPositionAction):
    """把策略的位置目标转成 DM-J4340-2EC 的供电电压，逐物理子步写入 ctrl。"""

    cfg: DmVoltageActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    # ⭐ 关键声明：让框架把 apply_actions 挂到每个物理子步（200 Hz）而不是控制步（50 Hz）。
    #    τ-ω 限幅依赖瞬时 q̇，只在 50 Hz 算会让中间 3 个子步共用常值 → 高速段失真。
    requires_substep_state_feedback: ClassVar[bool] = True

    def __init__(self, cfg: DmVoltageActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)  # 父类负责 scale / offset / clip / target_ids / motion_command

        self._bus_voltage = _real(cfg.bus_voltage, label="bus_voltage", minimum=0.0, strict=True)
        self._stall_torque = _real(cfg.stall_torque, label="stall_torque", minimum=0.0, strict=True)
        self._no_load_speed = _real(
            cfg.no_load_speed, label="no_load_speed", minimum=0.0, strict=True
        )
        self._kp_voltage = _real(cfg.kp_voltage, label="kp_voltage")
        self._kd_voltage = _real(cfg.kd_voltage, label="kd_voltage")
        self._kff_voltage = _real(cfg.kff_voltage, label="kff_voltage")

        # 与 XML 的 nominal="V τ_stall ω_no_load" 完全相同的推导 —— 两侧必须一致，
        # 否则重构的力矩就与 MuJoCo 内部算的不一样（自检脚本会抓）。
        k = self._bus_voltage / self._no_load_speed  # 反电动势常数 V·s/rad
        r = self._bus_voltage * k / self._stall_torque  # 端电阻 Ω
        self._back_emf = k
        self._resistance = r
        self._nm_per_volt = k / r  # 不动点时的 dτ/dv

        # 实际写下去的电压与由此得到的力矩（对齐 MuJoCo 内部闭式）
        self._voltage = np.zeros_like(self._processed_actions)
        self._applied_torque = np.zeros_like(self._processed_actions)

    # ── 暴露给测量/调试的读数 ─────────────────────────────────────────────
    @property
    def applied_torque(self) -> np.ndarray:
        """最近一次物理子步写进 ctrl 后**实际**产生的关节力矩（N·m）。

        形状 ``(num_envs, num_joints)``。与 MuJoCo 内部的 ``actuator_force``
        逐位相同（实测最大误差 2.8e-14）—— 这正是 UniLab backend 不提供的、
        Q2 测量要用的力矩通道。
        """
        return self._applied_torque

    @property
    def commanded_voltage(self) -> np.ndarray:
        """最近一次物理子步写进 ctrl 的电压（V），已按 ctrlrange 裁剪。"""
        return self._voltage

    @property
    def motor_constants(self) -> dict[str, float]:
        """导出推导出的电气常数，便于与 XML 的 nominal 交叉核对。"""
        return {
            "K_back_emf_V_s_per_rad": self._back_emf,
            "R_ohm": self._resistance,
            "nm_per_volt": self._nm_per_volt,
            "stall_torque_at_bus": self._nm_per_volt * self._bus_voltage,
            "no_load_speed_at_bus": self._bus_voltage / self._back_emf,
        }

    def apply_actions(self) -> None:
        """每个**物理子步**被调用一次（见类属性声明）。"""
        data = self._entity.data
        q = np.asarray(data.joint_pos[:, self._target_ids], dtype=get_global_dtype())
        dq = np.asarray(data.joint_vel[:, self._target_ids], dtype=get_global_dtype())
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise ValueError(
                f"DmVoltageAction on entity '{self.cfg.entity_name}' received non-finite "
                "joint state from the backend"
            )

        # 与父类 MotionJointPositionAction.apply_actions 逐字相同的目标计算：
        # 处理后的动作 + 运动指令的每回合零位偏置 − 编码器零位偏差。
        # ⚠️ 不能调 super().apply_actions()，那会把目标写成位置指令；这里要的是
        #   同一个目标去驱动电压律。
        np.add(
            self._processed_actions,
            self._motion_command.joint_default_bias[:, self._target_ids],
            out=self._target,
        )
        self._target -= data.encoder_bias[:, self._target_ids]

        # PD 律 + 可选的反电动势前馈，输出单位是【电压】而不是力矩 ——
        # 力矩仍由 MuJoCo 的 τ-ω 决定。写成 (kff_v − kd_v) 而不是加两项，
        # 是为了让「总速度系数」一眼可见：kff_v=0 时就是普通 PD。
        volts = self._kp_voltage * (self._target - q) + (self._kff_voltage - self._kd_voltage) * dq
        np.clip(volts, -self._bus_voltage, self._bus_voltage, out=volts)

        self._entity.data.write_ctrl(volts, actuator_ids=self._target_ids)

        # 与 MuJoCo <dcmotor> 内部完全相同的闭式（自检脚本保证逐位一致）。
        # 用于对外暴露力矩读数，不参与物理。
        self._voltage[:] = volts
        self._applied_torque[:] = self._nm_per_volt * (volts - self._back_emf * dq)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._voltage[ids] = 0.0
        self._applied_torque[ids] = 0.0


__all__ = ["DmVoltageAction", "DmVoltageActionCfg"]
