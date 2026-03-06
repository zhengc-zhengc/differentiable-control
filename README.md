# Differentiable Control

基于 PyTorch 的可微分自动驾驶控制器调参框架。将传统 C++ 控制器（横向多点预瞄 + 纵向级联 PID）用 `nn.Module` 复现，结合自行车运动学模型进行 50Hz 闭环仿真，通过 BPTT 反向传播梯度实现端到端参数优化。

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
docs/                   # 设计文档
```
