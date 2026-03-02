# 可微控制器开发指南

## 目标

将 `output/` 中提取的 C++ 控制器用 PyTorch 复现，搭配自行车模型作为被控对象，通过设计 loss 函数用梯度下降进行控制器调参。

## 目录结构

- `model/` — 自行车模型（被控对象）
- `controller/` — PyTorch 控制器（从 `output/src/mlp/control/` 翻译）
- `optim/` — loss 设计 + 梯度下降调参
- `tests/` — 单元测试

## 参考文档

- `docs/controller_spec.md` — 控制器完整规格（逐字转录）
- `docs/tunable_params_analysis.md` — 可调参数分析（信号流 + 参数表）
- `docs/v4_control_analysis_summary.md` — 控制模块设计总结
- `output/src/mlp/control/` — C++ 代码骨架（翻译源）

## 开发环境

```bash
# 从项目根目录
uv venv && source .venv/Scripts/activate && uv pip install torch
```
