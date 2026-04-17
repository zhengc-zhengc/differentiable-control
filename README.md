# Differentiable Control — 可微分自动驾驶控制器调参框架

基于 PyTorch 的自动驾驶控制器可微调参框架。将工业级 C++ 控制器（横向多点预瞄 + 纵向级联 PID）用 `nn.Module` 复现，搭配可插拔的车辆模型进行 50Hz 闭环仿真，通过 BPTT（Backpropagation Through Time）反向传播梯度实现端到端参数优化。

## 项目概述

传统控制器参数靠人工整定（PID 调参、查找表标定），耗时且依赖工程师经验。本项目将整个闭环仿真构建为可微计算图，使控制器参数可以用梯度下降自动优化：

```
                        ┌─────── 计算图（可微）────────┐
                        │                              │
参考轨迹 ──→ 控制器(nn.Module) ──→ 车辆模型 ──→ 车辆状态 ──→ 下一步
                 ↑                                     │
                 │                              tracking loss
            nn.Parameter                        (横向/航向/速度误差)
            (查找表y值, PID增益)                        │
                 ↑                                     │
                 └──────── backward (BPTT) ────────────┘
```

## 控制器架构

调参对象是一套工业级自动驾驶控制器，横向→纵向串行执行，50Hz 控制循环。

### 横向控制器（LatControllerTruck）

多点预瞄 + 曲率前馈，纯比例控制（无积分器）：

```
参考轨迹 ──→ 三点预瞄（当前/近/远） ──→ 横向误差 + 航向误差
                                            │
                              反馈项（120°/s 限速） + 前馈项（165°/s 限速）
                                            │
                                    合并输出（300°/s 限速） ──→ 方向盘转角
```

- 核心参数：8 张速度查找表（T1–T8），控制预瞄距离、收敛时间、预瞄点位置等
- 可微参数：T2（预瞄时间）、T3（收敛时间）、T4（角速度预瞄）、T6（远预瞄时间）的 y 值
- 固定参数：T1/T5/T7/T8 和 kLh（物理约束/安全限制）

### 纵向控制器（LonController）

级联 PID + 扭矩输出层，对齐 C++ `CalFinalTorque` 公式：

```
站位误差 ──→ 站位 PID(kp, ki) ──→ 期望速度
                                       │
                  低速 PID(kp, ki) / 高速 PID(kp, ki)    ← switch_speed 切换
                                       │
                  加速度限幅（L1-L5 查找表） ──→ 加速度指令
                                       │
              扭矩输出层（风阻+滚阻+惯性力+P 补偿）× 车轮半径 ──→ 车轮总扭矩
              （kinematic 跳过这层，直接吃加速度；其他 plant 吃扭矩）
```

- 可微参数：7 个 PID 标量（station/low_speed/high_speed 的 kp、ki + switch_speed）
- 固定参数：5 张物理限制查找表 L1–L5（动力/制动/舒适性约束）+ 扭矩公式 10 个物理常数（车质量/风阻/滚阻系数/车轮半径等，对齐真实 C++ 卡车参数）

## 关键技术

- **Straight-Through Estimator (STE)**：速率限制器 forward 用硬限幅，backward 无条件传梯度
- **Smooth 近似**：条件分支用 `smooth_step` 平滑过渡
- **TBPTT（截断 BPTT）**：将长仿真截断为 k 步片段反向传播
- **双模式验证**：训练用平滑近似，验证用原始硬限幅
- **Per-trajectory loss 归一化**：防止高 loss 轨迹主导训练方向

## 项目结构

```
├── sim/                                # 可微控制主目录
│   ├── common.py                       #   基础运算：lookup1d, smooth_clamp, rate_limit (STE), PID, IIR
│   ├── config.py                       #   YAML 配置加载/保存，查找表解析
│   ├── sim_loop.py                     #   50Hz 闭环仿真主循环（lat→lon→vehicle step）
│   ├── run_demo.py                     #   可视化 Demo：跑多种轨迹并生成结果图
│   ├── health_check.py                 #   一键体检：pytest + 基线性能 + 梯度健康检查
│   │
│   ├── model/                          #   车辆模型 + 轨迹生成
│   │   ├── vehicle.py                  #     BicycleModel — 运动学自行车模型 (x,y,yaw,v)，吃加速度
│   │   ├── dynamic_vehicle.py          #     DynamicVehicle — 6-DOF 动力学 V1（RK4），吃车轮扭矩
│   │   ├── dynamic_vehicle_v2.py       #     VehicleDynamicsV2 — 6-DOF 动力学 V2（逐轮胎力 + Euler）
│   │   ├── hybrid_dynamic_vehicle.py   #     HybridDynamicVehicle — V1 机理模型+MLP（旧版，保留兼容）
│   │   ├── generic_hybrid_vehicle.py   #     GenericHybridVehicle — 通用混合车辆（checkpoint 驱动 MLP）
│   │   ├── truck_trailer_vehicle.py    #     TruckTrailerVehicle — 牵引车+挂车双体动力学适配器
│   │   ├── truck_trailer_dynamics.py   #     底层 TruckTrailerNominalDynamics + MLPErrorModel（来自 truckdynamicmodel 上游拷贝）
│   │   ├── vehicle_factory.py          #     create_vehicle() — 根据配置创建车辆模型
│   │   └── trajectory.py               #     轨迹生成（8标准类型×6速度段）+ 变速剖面 + TrajectoryAnalyzer
│   │
│   ├── controller/                     #   控制器 (nn.Module)
│   │   ├── lat_truck.py                #     横向控制器：多点预瞄 + 曲率前馈
│   │   └── lon.py                      #     纵向控制器：级联 PID + 加速度限幅
│   │
│   ├── optim/                          #   训练与验证
│   │   ├── train.py                    #     可微调参训练（--trajectories 接收类型名自动展开全速度段）
│   │   └── post_training.py            #     训练后自动化：49 场景对比图、参数变化图、实验日志
│   │
│   ├── configs/
│   │   ├── default.yaml                #     默认参数（控制器 + 各车辆模型 + 仿真配置）
│   │   ├── tuned/                      #     调参结果 YAML（commit hash + 时间戳命名）
│   │   └── checkpoints/                #     MLP 权重文件 (.pth)
│   │
│   ├── results/
│   │   └── baseline/                   #     基线结果图（按被控对象分目录）
│   │
│   ├── tests/                          #     pytest 测试（200+ 用例）
│   └── learn/                          #     学习笔记
│
├── docs/                               # 设计文档
│   ├── controller_spec_v2.md           #   控制器完整规格（含纵向扭矩模型/坡度估计）
│   ├── tunable_params_analysis.md      #   可调参数分析
│   ├── bptt_gradient_explosion_analysis.md  # BPTT 梯度爆炸分析
│   ├── controller_reproduction_workflow.md  # 新控制器可微复现的标准 5 阶段流程
│   └── plans/                          #   设计文档与实现计划
│
└── requirements.txt                    # 依赖：torch, numpy, matplotlib, pyyaml, pytest
```

## 环境搭建

```bash
git clone https://github.com/zhengc-zhengc/differentiable-control.git
cd differentiable-control

uv venv && source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate                # Linux/Mac
uv pip install -r requirements.txt
```

## 快速开始

```bash
cd sim

# 1. 运行测试
python -m pytest tests/ -q

# 2. 可视化 Demo（推荐 truck_trailer：牵引车+挂车双体）
python run_demo.py --plant truck_trailer --save --no-show

# 3. 训练（可微调参）
python optim/train.py --plant truck_trailer --epochs 6

# 4. 独立验证
python optim/post_training.py --config configs/tuned/xxx.yaml --plant truck_trailer
```

## 两种运行模式

### 1. 非可微仿真（测试/基线评估）

控制器行为与原始 C++ 实现一致（`differentiable=False`），用于评估和对比。

```bash
cd sim

# 可视化 Demo
python run_demo.py --plant truck_trailer --save --no-show

# 加载调参结果对比
python run_demo.py --plant truck_trailer --config configs/tuned/xxx.yaml --save --no-show

# 一键体检
python health_check.py
```

### 2. 可微调参训练

构建可微计算图，通过 BPTT 优化控制器参数（`differentiable=True`）。每条轨迹算完立即 backward 释放计算图（per-traj backward），内存占用稳定不随 epoch 增长。

```bash
cd sim

# 默认全量训练：8 种轨迹类型 × 6 速度段 = 48 条轨迹
python optim/train.py --plant truck_trailer --epochs 6

# 指定轨迹类型（自动展开到全部 6 个速度段）
python optim/train.py --plant truck_trailer --trajectories lane_change clothoid_decel --epochs 10

# warm-start：从上次调参结果继续训练
python optim/train.py --plant truck_trailer --config configs/tuned/xxx.yaml --epochs 6
```

训练完成后自动生成：loss 曲线、分轨迹 loss 分项、baseline vs tuned 对比图（49 场景）、参数变化热力图、实验日志。所有产物保存到 `results/training/{plant}/{timestamp}/`。

## 训练 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--plant` | kinematic | 被控对象（见下方「车辆模型」） |
| `--epochs` | 100 | 训练轮数（推荐 6 快速迭代） |
| `--trajectories` | 全部 8 种 | 轨迹类型名，空格分隔，自动展开到 6 速度段 |
| `--config` | None | 初始参数配置路径（warm-start） |
| `--lr` | 5e-2 | 主学习率（PID 增益等标量参数） |
| `--lr-tables` | 5e-2 | 查找表 y 值学习率 |
| `--tbptt-k` | 150 | TBPTT 截断窗口（步数，150 步 = 3 秒） |
| `--grad-clip` | 10.0 | 梯度范数裁剪阈值 |
| `--sim-length` | None | 仿真距离限制 (m)，None 为全长 |
| `--w-lat` | 10.0 | 横向误差 loss 权重 |
| `--w-head` | 8.0 | 航向误差 loss 权重 |
| `--w-speed` | 3.0 | 速度误差 loss 权重 |

## 车辆模型（被控对象）

### 配置文件

所有车辆参数和控制器参数集中在一个 YAML 文件中：**`sim/configs/default.yaml`**。

训练和验证脚本默认加载 `default.yaml`，通过 `--plant` 参数切换被控对象。`--plant` 的值会设置 yaml 中 `vehicle.model_type` 字段，系统据此选择对应的车辆模型类和参数段：

```bash
# --plant 指定被控对象，系统自动选择 default.yaml 中对应的参数段
python optim/train.py --plant hybrid_v2 --epochs 6
```

### 可用模型

| `--plant` 值 | 模型类 | 说明 |
|------|------|------|
| `kinematic` | BicycleModel | 运动学自行车模型（默认，训练最快） |
| `dynamic` | DynamicVehicle | 6-DOF 动力学 V1（RK4 积分） |
| `hybrid_dynamic` | HybridDynamicVehicle | V1 动力学 + MLP 残差（旧版） |
| `hybrid_v2` | GenericHybridVehicle | V2 动力学 + checkpoint 驱动 MLP |
| **`truck_trailer`** | **TruckTrailerVehicle** | **牵引车+挂车双体动力学（12D 状态 + RK4 + MLP 残差，本地实现，无外部依赖）** |

**接口约定**：
- `kinematic` 吃加速度：`step(delta, acc=...)`
- 其他动力学/混合 plant 吃车轮总扭矩：`step(delta, torque_wheel=...)`
- 切换由 `sim_loop.py` 自动判断，控制器无感知
- 所有 plant 对外暴露 `x, y, yaw, v` 均为**后轴坐标**（控制器约定）

### hybrid_v2 配置详解

`hybrid_v2` 是推荐的高保真被控对象，由 **base 动力学 + 可选 MLP 残差修正** 组成。使用 `--plant hybrid_v2` 时，系统读取 `default.yaml` 中以下配置：

```yaml
# ---- sim/configs/default.yaml 中的相关配置 ----

vehicle:
  model_type: hybrid_v2
  # ↓ hybrid_v2 专属字段（--plant hybrid_v2 时自动填充默认值，也可在 yaml 中显式设置）
  base_model: dynamic_v2                # base 模型注册名 → 映射到 VehicleDynamicsV2 类
  params_section: dynamic_v2_vehicle    # 车辆物理参数在 yaml 中的 section 名
  checkpoint_path: configs/checkpoints/best_error_model_v2.pth  # MLP 权重路径（相对于 sim/）

# base_model: dynamic_v2 对应的物理参数段（参数必须与 MLP 训练时一致）
dynamic_v2_vehicle:
  mass: 2440.0
  Iz: 9564.8
  lf: 1.354
  lr: 1.446
  wheel_radius: 0.329
  steer_ratio: 16.39
  # ... 其他参数
```

**字段说明：**
- `model_type`: 决定系统使用哪个车辆类。`hybrid_v2` 对应 `GenericHybridVehicle`
- `base_model`: 纯物理动力学模型的注册名，在 `vehicle_factory.py` 的 `_BASE_MODEL_REGISTRY` 中查找
- `params_section`: 指向 yaml 中存放车辆物理参数的 section
- `checkpoint_path`: MLP 权重文件路径。**不设置则只使用纯 base 动力学，无 MLP**

MLP 的网络结构（层数、激活函数、输入特征、归一化参数）全部从 `.pth` 文件的元数据中自动读取，不需要在 yaml 中配置。

### truck_trailer 配置详解

`truck_trailer` 是牵引车+挂车双体动力学。底层模型（`TruckTrailerNominalDynamics` + `MLPErrorModel`）和 MLP checkpoint 都已**本地化**在本仓库，无需外部依赖：

- 动力学代码：`sim/model/truck_trailer_dynamics.py`（来自 [`mutespeaker/truckdynamicmodel`](https://github.com/mutespeaker/truckdynamicmodel) 上游 `base_model.py + model_structure.py` 的拷贝）
- MLP 权重：`sim/configs/checkpoints/truck_trailer_error_model.pth`

```yaml
vehicle:
  model_type: truck_trailer

truck_trailer_vehicle:
  # 车辆物理参数（与上游 BASE_MODEL_PARAMS 对齐）
  m_t: 9300.0          # 牵引车质量 (kg)
  L_t: 4.475           # 牵引车轴距 (m)
  a_t: 3.8             # 前轴到质心距离 (m)
  m_s_base: 15004.0    # 挂车基础质量 (kg)
  L_s: 8.0             # 挂车长度 (m)
  c_s: 4.0             # 挂车质心到铰接点距离 (m)
  wheel_radius: 0.5    # 车轮半径 (m)
  steering_ratio: 16.39
  # ... 其他空气动力 / 轮胎 / 铰接参数
  default_trailer_mass_kg: 15004.0   # 运行时挂车质量；改为 0 切到无挂车模式
  checkpoint_path: configs/checkpoints/truck_trailer_error_model.pth  # MLP 残差权重
```

**关键事实：**
- 内部维护 12D 状态（牵引车质心 6 + 挂车质心 6），对外仅暴露**牵引车后轴** x/y/yaw/v
- Base 用 RK4 积分（与上游 MLP 训练时一致）
- 挂车质量可通过 yaml 切换（`< 1.0 kg` 自动进入无挂车模式）
- 上游若有更新，需手动同步 `truck_trailer_dynamics.py`（文件头标注了上游版本）

### 新增被控对象

| 场景 | 操作 |
|------|------|
| 更换 MLP checkpoint（同 base 模型） | 替换 `.pth` 文件 + 改 yaml 中 `checkpoint_path`，**零代码改动** |
| MLP 结构变化（层数/激活函数/输入特征） | 同上，**零代码改动**（结构从 checkpoint 自动重建） |
| 新 base 动力学模型 | 1. 在 `sim/model/` 下写新的 `nn.Module`（实现 `forward(state, control, dt) → next_state`） 2. 在 `vehicle_factory.py` 的 `_BASE_MODEL_REGISTRY` 注册 3. 在 `default.yaml` 添加物理参数段 |

> **前后轴约定**：所有被控对象的 `x`、`y` 属性必须输出**后轴坐标**。内部动力学可使用任意参考点（前轴/质心），坐标转换在 vehicle 内部完成。

## 轨迹系统

训练和验证使用统一的轨迹类型集，每种类型在 6 个速度段（5/18/25/35/45/55 kph）都有对应版本。

### 恒速轨迹（5 种）

| 类型名 | 说明 |
|--------|------|
| `lane_change` | 余弦单换道（C2 连续） |
| `double_lc` | 双换道（左→右） |
| `clothoid_left` | 左转 clothoid 弯道（渐变曲率过渡） |
| `clothoid_right` | 右转 clothoid 弯道 |
| `s_curve` | S 弯（左弧→右弧） |

### 变速轨迹（3 种）

| 类型名 | 说明 |
|--------|------|
| `combined_decel` | 直线→弯道→直线，弯前自动减速/弯后加速 |
| `clothoid_decel` | clothoid 弯道 + 弯前减速/弯后加速 |
| `lc_accel` | 换道 + 梯形加减速（±15% 巡航速度波动） |

### 特殊轨迹

| 类型名 | 说明 |
|--------|------|
| `park_route` | 园区综合路线（仅验证时使用） |

## 独立验证

```bash
cd sim

# 全量验证（49 场景 = 8 类型 × 6 速度段 + park_route）
python optim/post_training.py --config configs/tuned/xxx.yaml --plant truck_trailer

# 指定类型验证
python optim/post_training.py --config configs/tuned/xxx.yaml --trajectories lane_change clothoid_decel

# 验证输出：5 种对比图 + 训练摘要仪表板 + 实验日志
```

## 文档

- [`docs/controller_spec_v2.md`](docs/controller_spec_v2.md) — 控制器完整规格（含纵向扭矩模型/坡度估计）
- [`docs/controller_reproduction_workflow.md`](docs/controller_reproduction_workflow.md) — 新控制器可微复现的标准 5 阶段流程
- [`docs/tunable_params_analysis.md`](docs/tunable_params_analysis.md) — 可调参数分析
- [`docs/bptt_gradient_explosion_analysis.md`](docs/bptt_gradient_explosion_analysis.md) — BPTT 梯度爆炸分析
- [`docs/plans/2026-04-15-torque-output-layer-design.md`](docs/plans/2026-04-15-torque-output-layer-design.md) — 纵向扭矩输出层设计
- [`docs/plans/`](docs/plans/) — 其他设计文档与实现计划
