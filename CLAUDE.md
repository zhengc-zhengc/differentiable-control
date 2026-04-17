# CLAUDE.md

本仓库包含两个阶段的工作：
1. **代码提取**（`analysis/`、`output/`、`tools/`）：从屏幕录像中提取自动驾驶控制模块的 C++ 代码结构
2. **可微控制**（`sim/`）：将提取的控制器用 PyTorch 复现（nn.Module），基于自行车模型进行可微调参

> **新控制器可微复现的标准流程见 [`docs/controller_reproduction_workflow.md`](docs/controller_reproduction_workflow.md)**——从截图到 tuned YAML 的 5 阶段工作流 + 踩坑 catalog。

## Target Code Architecture（控制模块架构概要）

目标项目是自动驾驶的**控制模块**，50Hz 控制循环，横向→纵向串行执行。已复现横向（LatControllerTruck，重卡）和纵向（LonController，含扭矩输出层），电拖横向（LatController）未复现。

> 控制器算法细节与可微实现见 `sim/CLAUDE.md`；完整规格见 `docs/controller_spec_v2.md`；参数分析见 `docs/tunable_params_analysis.md`

## Environment Setup

```bash
uv venv && source .venv/Scripts/activate && uv pip install opencv-python torch
```

## Project Layout

- `sim/` — **可微控制主目录**（PyTorch 控制器 + 自行车模型 + 训练，详见 `sim/CLAUDE.md`）
- `docs/` — 控制器文档（controller_spec、tunable_params_analysis、设计文档等）
- `docs/plans/` — 设计文档与实现计划（按日期命名）
- `ppt/` — 进展报告与 PPT 生成脚本
- `output/src/mlp/control/` — 从录屏截图推测的 C++ 代码骨架（仅供参考）
- `analysis/` — 录屏分析中间产物
- `tools/` — 抽帧 / 交叉验证工具
- `recordings/` — 原始录屏（.gitignore 排除）
- `.worktrees/` — git worktree 工作目录（feature/experiment 分支隔离开发，.gitignore 排除）

## Git 提交规范

- commit message 使用**中文**（专有名词和变量名除外）
- 一个完整任务的多个步骤应**合并为一个 commit**，不拆成碎片
- 前缀：`[sim]` 可微控制 / `[extract]` 录屏提取 / `[docs]` 纯文档
- 每次提交应关联：**代码 + 设计文档**（`docs/plans/`）**+ 结果图**（`sim/results/`）
- 代码框架变更（新增/删除/重命名模块、目录结构调整）时，**同步更新相关 CLAUDE.md**
- 每次完成代码和核心文档的改动后，**必须主动 commit**，不要等用户提醒
- 每次 commit 后，**同步 push 到远端**

## 可视化规范

- 图表标题、坐标轴标签、图例使用**中文**
- 每个子图都要有**图例**
- 结果图保存到 `sim/results/`，纳入 git

## 沟通风格

- 解释机制时用**自然语言直接描述物理/数学原理**，尽量少用代码黑话（变量名、类名、函数名、PyTorch 术语堆砌）
- 能用一句"把总扭矩平分到两个后轮"说清楚，就不要只贴 `torque_rear = torque_wheel / 2.0`
- 先讲"这件事在做什么、为什么这么做"，再在必要时引用代码位置佐证
- 回复避免冗余总结与套话，直奔结论
