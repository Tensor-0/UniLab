# 交接：unisim-core 1.1.4 → 1.7.2 迁移（2026-09-20）

> ⚠️ **本地实验的产物，未提交、未推上游。** 环境当前就在 1.7.2 上。
> 回滚见文末。

## 一、为什么做

`unisim-core` 停在 **1.1.4**（2026-09-08），而：

- PyPI 已到 **1.7.2**（09-19）
- 上游 `origin/main` 的 `pyproject.toml` 已经要求 `unisim-core>=1.4.2` +
  `unilab-rl==1.2.1` + `mjbatch-uni~=0.2.1`
- **UniLab #1602**（OPEN）记录：「unisim 1.4.0→1.5.1 后 MuJoCo collector 吞吐下降 19%–43%，
  主要耗时增量出现在**状态更新阶段**」；根因定位在上游 `unilabsim/unisim#142`
  （`_sync_tracked_body_state` 逐 env 串行 ctypes）

## 二、⭐ 结果：**+14.2%**（实测）

同一个 benchmark，512 envs / warmup 30 / 300 iters / 3+6 个独立进程：

| phase | 1.1.4 中位 | 1.7.2 中位 | 变化 | 核数 |
|---|---|---|---|---|
| backend_step | 3.63 | **3.08** | **−15.2%** | 16.59 → 19.37 |
| update_state | 1.85 | **1.58** | **−14.3%** | 1.10 → 2.28 |
| reset_done | 1.88 | **1.81** | −4.0% | 1.41 → 2.26 |
| **TOTAL** | 7.54 | **6.60** | **−12.5%** | |
| **steps/s** | 67,943 | **77,573** | **+14.2%** | |

- 两件事同时变好：**干得更少**（各相都降 15%）+ **串行段并行更好**（update_state 1.10 → 2.28 核）
- 机制对得上：1.7.2 的 `_sync_tracked_body_state` 已经是批量版
  （`self._pool.refresh_sensor_ranges(rows, ...)`），逐 env ctypes 循环降级为 fallback；
  **1.1.4 里根本没有这个函数**
- ⚠️ 1.7.2 那组数据分两簇（~78k × 4 / ~69k × 2），逐跑看是**系统级波动**
  （慢的两跑所有相位都慢，含 physics），不是相位退化
- **保守下界**：最慢的 1.7.2（68,689）仍 **+0.9%** 于最快的 1.1.4（68,069）

### 功能验证（dm10 + mujoco，我们唯一在跑的路径）

| 项 | 结果 |
|---|---|
| smoke（create_env + init_state + 40 步） | ✅ |
| 250 轮真实训练 | ✅ |
| 训练后回放录像 | ✅ 200 帧正常出片 |

## 三、这不是版本号改动，是**迁移**——改了 5 处

| # | 断在哪 | 怎么修 |
|---|---|---|
| 1 | 原生批处理引擎换了：`mujoco-uni-runtime 0.5.0` → **`mjbatch-uni 0.2.2`** | `pyproject.toml` 的 `mujoco` extra；顺带删掉 pybind11/wheel/`no-build-isolation-package` |
| 2 | `unisim.dr.types` 删了 `GeomSizeOverride` / `InitRandomizationPlan` / `ModelVariantSpec`，**语义也变了**（改 geom 尺寸 → 在整模型文件间选） | **新增 `src/unilab/dr/_compat.py`**：响亮 shim（照用即抛），`dr/__init__.py` + `dr/provider.py` 改走它 |
| 3 | 所有后端删了 `apply_init_randomization` | dm10 走不到（provider 默认返回 None）；**sharpa 会中招** |
| 4 | `run_playback(extra_data_getter=…)` → `debug_overlay_getter`（返回 `DebugPrimitive` 而非 ndarray） | `base/base.py` · `base/np_env.py` · `visualization/playback.py` · `scripts/train_rsl_rl.py` |
| 5 | isaacsim worker 删了私有 `_quat_rotate_wxyz` | 测试里改成守卫 import + skip |

## 四、⚠️ 测试影响与「未验证的部分」

同一套测试（`tests/base tests/envs tests/managers -k "not real_mujoco"`）：

| | 失败数 |
|---|---|
| 基线 1.1.4 | **25** |
| 升级 1.7.2 | **83** |
| **净新增** | **61** |

61 个的构成：

| 类型 | 数 | 说明 |
|---|---|---|
| `AttributeError` 私有属性改名 | 28 | 白盒测试戳 `_compiled_index` / `_entity_layout`→`_entity_roots` / `_execute_host_reset` 等 |
| genesis 后端能力边界 | 22 | 同一条 `NotImplementedError`；genesis 没装也不用 |
| isaacsim / isaacgym 依赖 | 4 | 需要专用 worker 解释器 |
| `dr/_compat.py` shim | 2 | `test_sharpa` 的两个 plan 构造测试 |
| 文案 / 字段变化 | 5 | 含 `BackendPlayCapabilities` 新增 `debug_overlay`；mjwarp 快照布局文案多了 `, (mocap_pos, mocap_quat)?` |

### ⚠️ 明确未验证的

1. **多 free-joint 任务**：`test_mujoco_root_layout_resolves_a_nonfirst_free_joint` 失败。
   dm10 只有一个 free joint（base），走不到那条分支。
2. **28 个私有属性改名说明上游重构了后端内部结构**。我们没跑过覆盖那些路径的任务
   （terrain、多实体、mocap、motrix、genesis、isaac 系）⇒ **那些后端的兼容性是未知的**。
   它们本来也不在本项目的支持范围内，但要有意识地知道这一点。
3. `Makefile` 里 `make mujoco MJ=<version>`（mujoco-uni-runtime 的 sdist 回退构建）
   **现在是死路径** —— 引用的是已删除的包。要重编译原生层得按 mjbatch-uni 的方式重写。

## 五、回滚

```bash
cd ~/UniLab
cp /tmp/unisim_upgrade_backup/uv.lock.1.1.4 uv.lock
cp /tmp/unisim_upgrade_backup/pyproject.toml.bak pyproject.toml
# 源码改动（dr/_compat.py 等）也要一并撤掉，否则签名对不上：
git checkout -- src/unilab/dr/ src/unilab/base/base.py src/unilab/base/np_env.py \
                src/unilab/visualization/playback.py src/unilab/scripts/train_rsl_rl.py \
                tests/base/test_motrix_backend_options.py tests/base/test_genesis_backend.py \
                tests/base/test_isaacsim_backend.py
rm -f src/unilab/dr/_compat.py
uv sync --locked --extra mujoco --extra mjwarp
```

## 六、⚠️ 一个顺带发现（可能要单独处理）

**1.1.4 没有 `_sync_tracked_body_state`**，意味着那一版的 obs 传感器读数可能是
**滞后一子步**的（和上游 1.4.0 同样的"快但陈旧"路径，见 `unilabsim/unisim#142`）。
1.7.2 改成了**正确但更贵**的刷新（批量版把代价压回去了）。

⇒ **升级不只是性能问题，也换了 obs 语义。** 如果之前的策略是带着这个滞后训出来的，
换版本后**同一份 checkpoint 的输入分布会变**。这一点在上机前值得确认。
