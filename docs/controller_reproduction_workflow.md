# 控制器可微复现工作流

> 把 C++ 自动驾驶控制器从屏幕截图复现成 PyTorch 可微控制器、接入 `sim/` 管线、调参产出 tuned YAML 的标准流程。
>
> 本文档面向 AI agent 和人类协作者共同阅读。节奏是"讨论伙伴模式"：agent 按 Phase 走流程，每个检查点（标 **Q** 编号）必须停下来问用户，不能自行跳过。

---

## §0 概述与入口

### 适用场景

有一个新的 C++ 控制器（横向 / 纵向 / 其它），你有若干截图（代码骨架、参数表、流程图、文档），目标是：
1. 从截图重建控制器的行为
2. 用 PyTorch 写出可微版本
3. 接入现有 `sim/` 管线
4. 调参产出 tuned YAML

### 入口 prompt 模板

新 agent 会话里这样说即可：

> 参考 `docs/controller_reproduction_workflow.md` 从 `<截图目录路径>` 复现 `<控制器名>` 控制器。

### Agent 开场动作

1. 读根 `CLAUDE.md`
2. 读 `sim/CLAUDE.md`
3. 读本文档
4. 列出截图目录里的图片清单，进入 Phase 1

### 节奏规则（硬性）

- 每个 Phase 的节奏是：**步骤 → 停下来问人 → 产出物**
- 所有标 **Q 编号** 的位置是硬性检查点，agent **必须**停下来问用户，不能跳过、不能并行
- 每个 Phase 的产出物落盘后才能进入下一个 Phase
- 用户可以随时喊暂停、回退、修改上游决策——workflow 允许 Phase 之间来回走
- 遇到 §6 catalog 里覆盖的场景时，在对话里用 `见 §6 #N` 引用，不重复写规则

### 整体结构

| 编号 | 名称 | 产出 |
|------|------|------|
| §1 | Phase 1：截图理解 | `docs/controller_spec_<name>_draft.md` |
| §2 | Phase 2：边界定义 | `docs/controller_spec_<name>.md` |
| §3 | Phase 3：numpy 基线 | `sim/controller/<name>.py` 硬限幅版 + 测试 |
| §4 | Phase 4：可微化 | `differentiable=True` 分支 + 梯度健康报告 |
| §5 | Phase 5：接入 + 调参 | sim 完整集成 + `tuned_*.yaml` |
| §6 | 踩坑 catalog | 跨 Phase 引用的硬规则 |
| §7 | 文档关联 | 与项目其他文档的关系 |

---

## §1 Phase 1：截图理解

**目标**：把截图内容转成结构化 spec 草稿。**不做取舍**，只求把看到的全列出来。

**前置条件**：
- 用户已给出截图目录路径 + 控制器名
- agent 已读根 `CLAUDE.md` 和 `sim/CLAUDE.md`

**步骤**：
1. 列出截图清单，每张标类别（代码骨架 / 流程图 / 参数表 / 测试样例 / 文档 / 其它）
2. 逐张 OCR + 结构化抽取
3. 汇总到 `docs/controller_spec_<name>_draft.md`，至少包含：
   - 函数 / 方法入口（signature）
   - 输入信号清单（名称 + 推测物理含义）
   - 输出信号清单
   - 中间状态变量
   - 查找表清单（1D / 2D / 3D）
   - 魔法常量清单
   - 从截图能看出的 step 顺序或流程图
4. 在对话里给出简短摘要，不贴大段 spec 内容

**停下来问人**：
- **Q1.1** 列出识别到的所有变量 / 函数 / 常量的清单，问用户有没有漏的或识别错的（截图 OCR 在符号上容易出错，比如 `θ` vs `θ̇`、`ψ` vs `φ`）
- **Q1.2** 确认控制器命名 + Python 文件路径（默认 `sim/controller/<name>.py`）

**产出物**：
- `docs/controller_spec_<name>_draft.md`（粗 spec，Phase 2 会基于它精简）

---

## §2 Phase 2：边界定义（最关键的一步）

**目标**：把粗 spec 精简成"仿真要复现的版本"——定接口、定坐标、定单位、定跳过项。产出是 Phase 3-5 唯一权威参考。

**前置条件**：Phase 1 draft spec 已落盘；Q1.1 / Q1.2 已通过。

**步骤**：
1. 输入 / 输出信号对齐 sim 现有接口
   - 每个输入信号能否从 `sim/model/trajectory.py` 的 `TrajectoryPoint` 或 `vehicle` 的状态属性直接取到？
   - 取不到的要怎么处理（补信号 / 替代变量 / 标记 TODO）
2. 选坐标参考点（前轴 / 后轴 / 质心）——默认后轴，见 §6 #3
3. 定符号约定和单位（CCW+ / CW+、rad / deg、m/s / kph）——默认对齐现有控制器，见 §6 #4
4. 列 C++ 原实现每一个 step，**用户逐条拍板**（必复现 / 简化 / 跳过）
   - agent **只列清单，不提建议**
   - 这一步是整个工作流里最需要人类判断的地方
5. 把定稿写进 `docs/controller_spec_<name>.md`（精简版）

**停下来问人**（这个 Phase 问最多）：
- **Q2.1** 输入信号 + sim 对照表（列出每个信号能否取到、缺失的处理方案），请用户确认
- **Q2.2** 坐标参考点建议**后轴**，请用户确认（参考 §6 #3 的历史教训）
- **Q2.3** 列出 C++ 原实现的所有 step，**不给建议**，请用户逐条拍板必复现 / 简化 / 跳过
- **Q2.4** 符号约定与单位（建议对齐现有控制器），请用户确认
- **Q2.5** 输出信号接到 sim_loop 哪条路径？
  - 横向控制器：输出 `delta`（前轮转角）
  - 纵向控制器：输出 `torque_wheel` 或 `acc`
  - 其它类型的控制器此问题按需展开

**产出物**：
- `docs/controller_spec_<name>.md`（精简版 spec，Phase 3-5 唯一参考）

---

## §3 Phase 3：numpy 基线

**目标**：按精简 spec 写能跟得上直线轨迹的 `differentiable=False` 版本。

**前置条件**：`docs/controller_spec_<name>.md` 已定稿。

**步骤**：
1. 按 spec 写 `sim/controller/<name>.py`：
   - `__init__(cfg, differentiable=False)` 从 cfg 读参数
   - `compute(...)` 主方法
   - `reset_state()`
   - 先只实现 `differentiable=False` 分支，使用 `np.clip` / `np.sign` 等硬约束
2. 在 `sim/configs/default.yaml` 加参数块，初值用 C++ 原值
3. 写基础单元测试 `sim/tests/test_<name>.py`：
   - 构造 / reset 正常
   - 典型输入的输出值
   - 边界输入（零速度、大误差、限幅触发等）
4. 在 `sim_loop.py` 临时挂通，跑直线轨迹验稳（`lateral_error` / `speed_error` 不发散、不 NaN）
5. 记录基线 RMSE

**异常处理**：基线 RMSE 异常时，agent **先自查**下面三类低级错误再来问人：
- 单位错误（rad vs deg、m/s vs kph）
- 符号约定错误（CCW+ vs CW+）
- 坐标参考点错配（前轴 vs 后轴）

自查 1-2 轮仍定位不到再把现象报给用户。不要每次基线偏差就立刻问人。

**停下来问人**：
- **Q3.1** 参数结构设计（哪些是 1D / 2D 表、哪些是标量，命名和层级），请用户确认
- **Q3.2** 基线直线跟踪 RMSE 值 + agent 的判断（符合预期 / 异常）及后续动作提议（继续 / 回头对 spec / 排查具体项）

**产出物**：
- `sim/controller/<name>.py`（只含 numpy 分支 + 接口骨架）
- `sim/tests/test_<name>.py`（基础结构 + 输出正确性）
- `sim/configs/default.yaml`（新增参数段）

---

## §4 Phase 4：可微化

**目标**：加 `differentiable=True` 分支，梯度能从 loss 流到所有 `nn.Parameter`，BPTT 数值健康。

**前置条件**：Phase 3 numpy 基线稳定、测试通过。

**步骤**：
1. 扫代码找**硬点**（不可微操作）：
   - 硬分支（`if` / `else` 条件依赖 tensor 的值）
   - 硬限幅（`max` / `min` / `clip` / `clamp`）
   - 不连续（`sign` / `abs` at 0 / `round` / `floor`）
   - 查表（1D / 2D lookup）
   - `argmin` / `argmax`
2. 对每个硬点按 **§6 catalog** 选替换方案：
   - **先看能否化简**（例如 `sign(x)·min(|x|,L)` 直接是 `clamp`，见 §6 #1）
   - 不能化简再上近似
   - 硬 clamp → Straight-Through Estimator（见 §6 #5 / #6）
   - 条件分支 → `smooth_step` 混合（见 §6 #7）
   - `argmin` / `argmax` → `detach` 隔离（见 §6 #8）
   - 时间查表 → `query_by_time_differentiable`（见 §6 #10）
3. 加 `differentiable=True` 分支（通常是函数里的 `if self.differentiable:` 判断）
4. 跑梯度健康检查：短轨迹训练 1 步，检查每个 `nn.Parameter` 的 `grad.norm()`、是否 NaN / Inf

**停下来问人**：
- **Q4.1** 扫出的硬点 + agent 初拟的替换方案表格。**重点标出可能踩 §6 catalog 的点**（例如识别到 `sign()` 时要显式提 §6 #1，问用户是否能化简）
- **Q4.2** 梯度健康报告表格（参数名 / grad_norm / 状态 OK / WARN_ZERO / WARN_SMALL / ERROR）
  - 预期的零梯度（仅监控输出、常曲率轨迹上的时间预瞄参数）在报告里标出，不视为问题
  - 意外的零梯度 agent 去查链路（通常是某处 `detach` 或 numpy 运算切断了计算图）

**产出物**：
- `sim/controller/<name>.py` 追加 `differentiable=True` 分支
- 梯度健康报告（对话里贴，不存独立文件）

---

## §5 Phase 5：接入 + 调参

**目标**：正式接入 sim 管线、基线验证、产出 tuned YAML。

**前置条件**：Phase 4 梯度健康通过。

**步骤**：
1. **参数分类**（参考 `sim/CLAUDE.md` 的"控制器参数分类"表格范式）：
   - 物理常数 / 安全约束 / 硬边界 / 不参与反馈的参数 → `register_buffer`（见 §6 #2）
   - PID 增益 / 预瞄时间 / 可调查找表 y 值 → `nn.Parameter`
   - 所有 `nn.Parameter` 必须加投影约束（见 §6 #4）
2. 挂到 `sim/optim/train.py` 的 `DiffControllerParams`
3. `sim_loop.py` 正式集成（如果控制器类型或数据流有变化）
4. 更新 `sim/CLAUDE.md`：
   - 模块结构章节加新控制器
   - 参数分类表格加新控制器的可微 / 固定参数
   - 如与 spec 有差异写进"与 controller_spec_v2.md 的差异"章节
5. 全量 `pytest tests/` 通过
6. 基线可视化：`python run_demo.py --save --no-show`
7. **短训练验证管线**：
   ```
   python optim/train.py --epochs 1 --trajectories lane_change --sim-length 15
   ```
   目的是确认 loss 不发散、梯度不爆、pipeline 能跑完。
8. **中量训练验收**：3 种代表性轨迹类型（`lane_change` + `clothoid_left/right` + `s_curve`）× 6 速度段 = 18 条轨迹，跑完整 epoch 数
9. 如有必要上**完整训练**（8 类型 × 6 速度段 = 48 条）
10. 产出 tuned YAML + `post_training` 自动化产物

**停下来问人**：
- **Q5.1** 参数分类表（可微 / 固定 + 每项理由），请用户确认
- **Q5.2** 短训练 1 epoch loss 从 x 到 y，agent 判断流程通了，提议上中量训练，请用户确认
- **Q5.3** 中量 / 完整训练完：初始 loss / 最终 loss / V1 验证 N 场景横向改善百分比（N 取决于控制器类型），请用户判断是否达到验收标准、要不要 warm-start 继续

**产出物**：
- 完整 sim 集成（`sim/controller/<name>.py`、`sim/configs/default.yaml`、`sim/tests/test_<name>.py`、`sim/sim_loop.py` 如有修改、`sim/optim/train.py`、`sim/CLAUDE.md`）
- `sim/configs/tuned/tuned_<hash>_<timestamp>.yaml`
- `sim/results/training/<plant>/<timestamp>/`（loss 曲线、baseline vs tuned 对比、实验日志 YAML）

---

## §6 踩坑 catalog（跨 Phase 硬规则）

> Phase 里遇到这些场景时用编号引用此节（`见 §6 #N`），不在 Phase 说明里重复写规则。

### #1 `sign(x)·min(|x|,L)` 不要用 smooth_sign，直接 `clamp(x,-L,L)`

- **症状**：BPTT 训练早期梯度爆炸（`grad_norm` 冲到 1e6+），loss 值本身可能不爆（因为是标量平均掩盖）
- **原理**：`sign(x)·min(|x|,L)` 数学上恒等于 `clamp(x,-L,L)`。前者改成可微版会引入 `smooth_sign` 的大导数（`1/ε` 级），与 clamp 的导数（1 或 0）相差几个数量级，BPTT 链式乘几百步直接溢出
- **规则**：**先化简数学表达式，再选近似方式**，不要机械替换每一个 `sign`
- **参考**：`docs/bptt_gradient_explosion_analysis.md`

### #2 不参与反馈的参数设 `register_buffer`，不设 `nn.Parameter`

- **例子**：LatControllerTruck 的 T5（近预瞄点时间，仅作监控输出）、T8（轮胎侧滑物理属性）、kLh（铰接几何）；纵向的 L1-L5（动力系统物理极限 / 舒适约束）；`lon_torque` 段的 10 个物理常数
- **原理**：`nn.Parameter` 参与优化，梯度为零的参数让优化器白跑且易制造数值噪声；`register_buffer` 只保留值不优化
- **判断依据**：**这个参数的变动是否会通过反馈路径影响 loss**。不会的一律 buffer

### #3 坐标参考点默认后轴

- **历史背景**：动力学模型初版用前轴为参考点，弧线轨迹上 `lateral_error` 有系统性偏差（commit `58cc6ae` 修复）。现在所有 plant 对外暴露的 `x / y / yaw / v` 都是**后轴**坐标
- **规则**：
  - 新控制器计算 `lateral_error` / `heading_error` 基于后轴
  - 如果控制器内部需要前轴信息（例如自行车模型的前轮 `delta` 计算），在模块内部做转换
  - **对外接口一律后轴**

### #4 可微参数必须加投影约束

- **场景**：PID 增益、预瞄时间参数等
- **原因**：未约束时优化器会把 `ki` 推到负值（物理不合理、闭环发散）；`T2/T3/T4/T6` 等时间参数负值没有物理意义
- **规则**（实现在 `sim/optim/train.py` 的 projection step，已有参考）：
  - 所有 PID 增益：`clamp_(min=0)`
  - 所有时间预瞄参数（T2 / T3 / T4 / T6）：`clamp_(min=0)`
  - `switch_speed`：`clamp_(min=0.5, max=10)`
  - 查找表 y 值如有物理下界同样 clamp
  - 在 `with torch.no_grad():` 块内做

### #5 `rate_limit` 用 Straight-Through Estimator

- **场景**：控制器输出有变化率限制（`|Δu/Δt| ≤ L`）
- **规则**：前向走硬 clamp（被限幅的动作照常发生），反向梯度 `d/dx = 1`（把 clamp 当恒等映射回传）
- **实现位置**：`sim/common.py` 的 `rate_limit(..., differentiable=True)` 内部用 `_StraightThroughClamp`

### #6 硬 clamp 用 STE，不用 smooth 近似

- 同 #5 的思路，用于一般输出限幅
- 如果必须用 smooth 近似（某些场景 STE 会导致优化器震荡），用 `sim/common.py` 的 `smooth_clamp`，但 `temp` 参数要大（≥ 0.1），不能小（见 #1）

### #7 条件分支用 `smooth_step` 混合

- **例子**：`y = a if x > threshold else b` → `y = s * a + (1 - s) * b`，其中 `s = smooth_step(x, threshold, temp)`
- **规则**：`temp` 不能太小，否则梯度在过渡区间内过大，BPTT 链式乘几百步爆炸（同 #1）
- **实现位置**：`sim/common.py` 的 `smooth_step`

### #8 `argmin` / `argmax` 用 `detach` 隔离

- **场景**：查表时找最近点
- **规则**：
  ```python
  idx = dist.argmin()  # 不可微
  idx = idx.detach()   # 显式隔离
  y = table_y[idx]     # y 仍能对 table_y 求导
  ```

### #9 BPTT 梯度爆炸的判断与排查

- **判断阈值**：`grad_norm > 1e4` 警戒，`> 1e6` 爆炸。loss 值本身可能不爆（标量平均会掩盖）
- **排查顺序**：
  1. 查 smooth 近似的 `temp` / `width` 是否过小（#1、#7）
  2. 用有限差分验证真实梯度是否有界（有界则纯是数值问题，不是数学问题）
  3. 检查是否有硬点没化简直接上了 smooth
- **防范**：`clip_grad_norm_` 作为安全网，不作为主要手段。优先调 `temp`
- **参考**：`docs/bptt_gradient_explosion_analysis.md`

### #10 时间查表用 `query_by_time_differentiable`

- **场景**：时间预瞄参数（T2 / T4 / T6）表示"在参考轨迹上往前看多少秒"，查表涉及时间轴插值
- **规则**：用 `sim/model/trajectory.py` 的 `TrajectoryAnalyzer.query_by_time_differentiable` 保证对时间参数 T 可求导
- **注意**：T6 在**变曲率轨迹**（换道 / clothoid）上有梯度，在**常曲率轨迹**（纯圆弧）上梯度为零是预期（曲率不变、预瞄点再往前看也是同一个值）

### #11 Plant 适配器必须实现 `detach_state()`

TBPTT 训练每 `tbptt_k` 步截断梯度链，需要 plant 暴露 `detach_state()` 方法。现有 plant（`BicycleModel` / `DynamicVehicle` / `HybridDynamicVehicle` / `GenericHybridVehicle` / `TruckTrailerVehicle`）都实现了。新增 plant 必须跟上。

### #12 Checkpoint 路径放仓库内部

MLP 残差 checkpoint 放在 `sim/configs/checkpoints/`，`default.yaml` 的 `checkpoint_path` 用相对路径（相对 `sim/`）。**不要直接指向外部仓库**（仓库迁移或克隆时会失联）。参考 truck_trailer 的处理方式。

---

## §7 与现有项目文档的关系

### 本文档管什么，不管什么

**本文档管**：跨控制器的**通用流程**和**硬规则**。

**本文档不管**：
- 具体某个控制器的 spec —— 那是每次复现产出的 `docs/controller_spec_<name>.md`
- 控制器算法细节 —— 参考 `docs/controller_spec_v2.md`、代码本身
- 可微技术细节 —— 参考 `docs/bptt_gradient_explosion_analysis.md`、`sim/CLAUDE.md`

### 产出物落盘位置速查

| Phase | 产出物 | 路径 |
|-------|--------|------|
| Phase 1 | 粗 spec | `docs/controller_spec_<name>_draft.md` |
| Phase 2 | 精简 spec（Phase 3-5 权威） | `docs/controller_spec_<name>.md` |
| Phase 3 | 控制器 numpy 分支 | `sim/controller/<name>.py` |
| Phase 3 | 单元测试 | `sim/tests/test_<name>.py` |
| Phase 3 | 参数初值 | `sim/configs/default.yaml` 新增段 |
| Phase 4 | 可微分支 | `sim/controller/<name>.py` 追加 |
| Phase 5 | 模块 / 参数分类同步 | `sim/CLAUDE.md` |
| Phase 5 | 调参产物 | `sim/configs/tuned/tuned_<hash>_<timestamp>.yaml` |
| Phase 5 | 训练图 + 日志 | `sim/results/training/<plant>/<timestamp>/` |

### 参考文档索引

- 根 `CLAUDE.md` —— 项目顶层约定、git 规范、沟通风格
- `sim/CLAUDE.md` —— sim 模块结构、参数分类、梯度约束
- `docs/controller_spec_v2.md` —— 既有控制器的完整 C++ 规格（只读参考）
- `docs/bptt_gradient_explosion_analysis.md` —— 梯度爆炸案例分析
- `docs/tunable_params_analysis.md` —— 既有控制器的可调参数分类分析
- `docs/plans/*.md` —— 历次设计决策（按日期命名）

---

## 附录：两个参考实现

复现新控制器时可以参考这两个已走通的例子：

- **LatControllerTruck（横向，重卡）**：`sim/controller/lat_truck.py`
  - 可微：T2 / T3 / T4 / T6 时间预瞄参数
  - 固定：kLh / T1 / T5 / T7 / T8
- **LonController（纵向）**：`sim/controller/lon.py`
  - 可微：station_kp/ki、low_speed_kp/ki、high_speed_kp/ki、switch_speed
  - 固定：L1-L5 + `lon_torque` 段 10 个物理常数
  - 含扭矩输出层：`compute_torque_wheel(acc_cmd, v, a_actual)`

新控制器的参数分类、`nn.Parameter` 命名风格、差异化接口（横向 vs 纵向）等，直接参考这两个模块的实现。
