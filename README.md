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

级联 PID 架构，外环站位 → 内环速度：

```
站位误差 ──→ 站位 PID(kp, ki) ──→ 期望速度
                                       │
                  低速 PID(kp, ki) / 高速 PID(kp, ki)    ← switch_speed 切换
                                       │
                  加速度限幅（L1-L5 查找表） ──→ 加速度指令
```

- 可微参数：7 个 PID 标量（station/low_speed/high_speed 的 kp、ki + switch_speed）
- 固定参数：5 张物理限制查找表 L1–L5（动力/制动/舒适性约束）

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
│   │   ├── vehicle.py                  #     BicycleModel — 运动学自行车模型 (x,y,yaw,v)
│   │   ├── dynamic_vehicle.py          #     DynamicVehicle — 6-DOF 动力学 V1（RK4 积分）
│   │   ├── dynamic_vehicle_v2.py       #     VehicleDynamicsV2 — 6-DOF 动力学 V2（逐轮胎力 + Euler）
│   │   ├── hybrid_dynamic_vehicle.py   #     HybridDynamicVehicle — V1 机理模型+MLP（旧版，保留兼容）
│   │   ├── generic_hybrid_vehicle.py   #     GenericHybridVehicle — 通用混合车辆（checkpoint 驱动 MLP）
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
│   ├── tests/                          #     pytest 测试（190+ 用例）
│   └── learn/                          #     学习笔记
│
├── docs/                               # 设计文档
│   ├── controller_spec.md              #   控制器完整规格
│   ├── tunable_params_analysis.md      #   可调参数分析
│   ├── bptt_gradient_explosion_analysis.md  # BPTT 梯度爆炸分析
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

# 2. 可视化 Demo（默认 hybrid_v2 被控对象）
python run_demo.py --plant hybrid_v2 --save --no-show

# 3. 训练（可微调参）
python optim/train.py --plant hybrid_v2 --epochs 6

# 4. 独立验证
python optim/post_training.py --config configs/tuned/xxx.yaml --plant hybrid_v2
```

## 两种运行模式

### 1. 非可微仿真（测试/基线评估）

控制器行为与原始 C++ 实现一致（`differentiable=False`），用于评估和对比。

```bash
cd sim

# 可视化 Demo
python run_demo.py --plant hybrid_v2 --save --no-show

# 加载调参结果对比
python run_demo.py --plant hybrid_v2 --config configs/tuned/xxx.yaml --save --no-show

# 一键体检
python health_check.py
```

### 2. 可微调参训练

构建可微计算图，通过 BPTT 优化控制器参数（`differentiable=True`）。

```bash
cd sim

# 默认全量训练：8 种轨迹类型 × 6 速度段 = 48 条轨迹
python optim/train.py --plant hybrid_v2 --epochs 6

# 指定轨迹类型（自动展开到全部 6 个速度段）
python optim/train.py --plant hybrid_v2 --trajectories lane_change clothoid_decel --epochs 10

# warm-start：从上次调参结果继续训练
python optim/train.py --plant hybrid_v2 --config configs/tuned/xxx.yaml --epochs 6
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

通过 `--plant` 参数或 `default.yaml` 中的 `vehicle.model_type` 选择。所有模型对外接口一致：`step(delta, acc)` 输入前轮转角和加速度，`x, y, yaw, v` 输出后轴坐标。

| `--plant` 值 | 模型 | 说明 |
|------|------|------|
| `kinematic` | BicycleModel | 运动学自行车模型（默认，训练最快） |
| `dynamic` | DynamicVehicle | 6-DOF 动力学 V1（RK4 积分） |
| `hybrid_dynamic` | HybridDynamicVehicle | V1 动力学 + MLP 残差（旧版） |
| **`hybrid_v2`** | **GenericHybridVehicle** | **V2 动力学 + checkpoint 驱动 MLP（推荐）** |

### hybrid_v2：通用混合车辆

`hybrid_v2` 是推荐的高保真被控对象。它由两部分组成：

```
Base 动力学模型（纯物理方程）  +  MLP 残差修正（从 .pth checkpoint 加载）
          │                              │
     可选择不同版本                  结构自动从 checkpoint 元数据重建
     (dynamic_v2 等)               (层数/激活函数/输入特征/归一化)
```

**配置方式**（`default.yaml`）：

```yaml
vehicle:
  model_type: hybrid_v2           # 使用通用混合车辆
  base_model: dynamic_v2          # base 动力学模型名
  params_section: dynamic_v2_vehicle  # 车辆参数在 yaml 中的 section 名
  checkpoint_path: configs/checkpoints/best_error_model_v2.pth  # MLP 权重

# V2 动力学参数（必须与 MLP 训练时一致）
dynamic_v2_vehicle:
  mass: 2440.0
  Iz: 9564.8
  lf: 1.354
  lr: 1.446
  # ... 其他参数
```

**不提供 checkpoint_path** 则只使用纯 base 动力学，无 MLP 修正。

### 新增被控对象指南

如果需要接入新的被控对象，只需以下步骤：

| 场景 | 需要做什么 |
|------|-----------|
| 仅更换 MLP checkpoint（同 base 模型） | 替换 `.pth` 文件 + 修改 yaml 中 `checkpoint_path`，**零代码改动** |
| MLP 结构变化（层数/激活函数/输入特征） | 同上，**零代码改动**（MLP 结构从 checkpoint 元数据自动重建） |
| 新 base 动力学模型 | 1. 写新的 `nn.Module` 类（实现 `forward(state, control, dt) → next_state`） 2. 在 `vehicle_factory.py` 的 `_BASE_MODEL_REGISTRY` 中注册 3. 在 `default.yaml` 添加对应参数段 |

所有被控对象必须遵循统一接口：

```python
class AnyVehicle:
    def step(self, delta, acc): ...    # 前轮转角 (rad) + 加速度 (m/s²)
    x: Tensor      # 后轴 x 坐标
    y: Tensor      # 后轴 y 坐标
    yaw: Tensor    # 航向角
    v: Tensor      # 合速度
    def detach_state(self): ...        # TBPTT 梯度截断
```

> **前后轴约定**：内部动力学可使用任意参考点（前轴/后轴/质心），但 `x`、`y` 属性必须输出**后轴坐标**。坐标转换在 vehicle 内部完成，控制器和仿真循环只看到后轴坐标。

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
python optim/post_training.py --config configs/tuned/xxx.yaml --plant hybrid_v2

# 指定类型验证
python optim/post_training.py --config configs/tuned/xxx.yaml --trajectories lane_change clothoid_decel

# 验证输出：5 种对比图 + 训练摘要仪表板 + 实验日志
```

## 文档

- [`docs/controller_spec.md`](docs/controller_spec.md) — 控制器完整算法规格
- [`docs/tunable_params_analysis.md`](docs/tunable_params_analysis.md) — 可调参数分析
- [`docs/bptt_gradient_explosion_analysis.md`](docs/bptt_gradient_explosion_analysis.md) — BPTT 梯度爆炸分析
- [`docs/plans/`](docs/plans/) — 设计文档与实现计划
