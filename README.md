# Differentiable Control

基于 PyTorch 的可微分自动驾驶控制器调参框架。将传统 C++ 控制器（横向多点预瞄 + 纵向级联 PID）用 `nn.Module` 复现，结合自行车运动学模型进行 50Hz 闭环仿真，通过 BPTT 反向传播梯度实现端到端参数优化。

## 控制器概览

本项目调参的对象是一套工业级自动驾驶控制器，横向和纵向串行执行，50Hz 控制循环。

### 横向控制器（LatControllerTruck — 重卡）

多点预瞄 + 曲率前馈算法，纯比例控制（无积分器）：

```
参考轨迹 ──→ 三点预瞄（当前/近/远） ──→ 横向误差 + 航向误差
                                            │
                                     realTheta → targetTheta → targetCurvature
                                            │
                              反馈项（120°/s 限速） + 前馈项（165°/s 限速）
                                            │
                                    合并输出（300°/s 限速） ──→ 方向盘转角
```

- 核心参数：8 张速度查找表（T1–T8），控制预瞄距离、收敛时间、预瞄点位置等
- 其中 T2–T6 的 y 值为可微参数（`nn.Parameter`），T1/T7/T8 和 kLh 为物理约束固定参数

### 纵向控制器（LonController）

级联 PID 架构，外环站位 → 内环速度：

```
站位误差 ──→ 站位 PID(kp, ki) ──→ 期望速度偏移量
                                       │
                  ┌────────────────────┤
                  │                    │
            低速 PID(kp, ki)     高速 PID(kp, ki)    ← switch_speed=3.0m/s 切换
                  │                    │
                  └────────┬───────────┘
                           │
                  加速度限幅（L1-L5 查找表） ──→ 加速度指令
```

- 可微参数：7 个 PID 标量（station/low_speed/high_speed 的 kp、ki + switch_speed）
- 固定参数：5 张物理限制查找表 L1–L5（动力/制动能力、加速度变化率上下限）
- 横纵耦合：横向控制器的远预瞄曲率传递给纵向控制器，用于急弯减速判断（单向）

## 可微调参方法

传统控制器参数靠人工整定；本项目将整个闭环仿真构建为可微计算图，用梯度下降自动优化：

```
                        ┌─────── 计算图（可微）────────┐
                        │                              │
参考轨迹 ──→ 控制器(nn.Module) ──→ 自行车模型 ──→ 车辆状态 ──→ 下一步
                 ↑                                     │
                 │                              tracking loss
            nn.Parameter                        (横向/航向/速度误差)
            (查找表y值, PID增益)                        │
                 ↑                                     │
                 └──────── backward (BPTT) ────────────┘
```

**关键技术**：
- **Straight-Through Estimator (STE)**：速率限制器（rate limiter）forward 用硬限幅，backward 无条件传梯度，避免梯度消失
- **Smooth 近似**：条件分支（如低/高速 PID 切换）用 `smooth_step` 平滑过渡，温度参数需谨慎选择以防 BPTT 链式乘法导致梯度爆炸
- **TBPTT（截断 BPTT）**：将长仿真序列截断为 k 步片段反向传播，平衡长程梯度与数值稳定性
- **双模式验证**：训练用 `differentiable=True`（平滑近似），得到的参数用 `differentiable=False`（原始硬限幅）重新仿真验证

## 核心特性

- **可微控制器**：横向（LatControllerTruck）+ 纵向（LonController），支持速度查找表和 PID 参数的梯度优化
- **自行车模型**：前轮转向运动学模型，支持 differentiable/non-differentiable 双模式
- **多轨迹训练**：直线、圆弧、正弦、换道等工况，TBPTT 截断反向传播
- **训练可观测性**：梯度健康检查、参数快照、分轨迹 loss 明细、自动可视化

## 快速开始

```bash
uv venv && source .venv/Scripts/activate  # Windows
uv pip install -r requirements.txt

# 运行仿真演示
cd sim && python run_demo.py

# 训练（可微调参）
cd sim && python -m optim.train

# 运行测试
cd sim && pytest
```

## 项目结构

```
sim/                    # 可微控制主目录
├── model/              #   自行车模型 + 轨迹生成
├── controller/         #   横向/纵向控制器 (nn.Module)
├── optim/              #   训练循环 + 训练后可视化
├── configs/            #   参数配置 (YAML)
└── results/            #   仿真结果图
docs/                   # 设计文档（控制器规格、参数分析、梯度爆炸分析等）
```
