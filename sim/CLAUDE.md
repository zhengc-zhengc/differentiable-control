# sim/ — 可微控制仿真

## 概述

将 C++ 控制器用 PyTorch 复现（`nn.Module`），搭配运动学自行车模型，通过 BPTT 进行梯度调参。
所有模块支持 `differentiable=True/False` 开关：`True` 全程 tensor 运算可回传梯度；`False` 与 V1（numpy）行为一致。

## 模块结构

```
sim/
├── common.py              # 基础运算：lookup1d, smooth_clamp/sign, rate_limit (STE), PID, IIR
├── config.py              # YAML 加载/保存，table_from_config → (Tensor, Tensor)
├── sim_loop.py            # 50Hz 闭环仿真（lat→lon→vehicle step）
├── run_demo.py            # 可视化 Demo（--save --no-show --config）
├── model/
│   ├── vehicle.py         # BicycleModel (x,y,yaw,v)，dt=0.02s
│   └── trajectory.py      # 4 种轨迹生成 + TrajectoryAnalyzer（detached argmin）
├── controller/
│   ├── lat_truck.py       # LatControllerTruck (nn.Module, 8 表 + kLh)
│   └── lon.py             # LonController (nn.Module, 7 PID + 5 表)
├── optim/
│   └── train.py           # 可微调参：DiffControllerParams, tracking_loss, Adam
├── configs/
│   ├── default.yaml       # 默认参数
│   └── tuned/             # 调参结果 YAML（文件名含 commit hash + timestamp）
├── results/               # 结果图（纳入 git）
└── tests/                 # pytest 测试（132 项）
```

## 数据流

**仿真**：`trajectory` 生成参考轨迹 → `sim_loop` 每步调用 `lat_ctrl.compute()` → `lon_ctrl.compute()` → `vehicle.step()` → 记录 history

**训练**：`DiffControllerParams` 封装两个控制器（nn.Module） → `sim_loop(differentiable=True)` 构建计算图 → `tracking_loss` 汇总误差（横向/航向/速度/平滑度） → `backward()` → Adam 更新 nn.Parameter → `save_tuned_config` 导出 YAML

**梯度路径**：loss → history 中的 tensor → vehicle 状态传播 → 控制器输出 → 查找表 y 值 / PID 增益（nn.Parameter）。不可微操作的处理：rate_limit 用 Straight-Through Estimator，条件分支用 smooth_step 混合，argmin 用 detach 隔离。

**验证**：训练使用平滑近似（`differentiable=True`），得到的参数必须用原始硬限幅控制器（`differentiable=False`，即 V1 路径）重新跑仿真验证，确保参数在真实限幅逻辑下同样有效。`run_demo.py --config` 默认即为 V1 路径。

## 常用命令

```bash
cd sim/
python -m pytest tests/ -q                                  # 跑测试
python run_demo.py --save --no-show                         # 生成结果图
python run_demo.py --config configs/tuned/xxx.yaml          # 加载调参结果
python optim/train.py --epochs 50 --trajectories circle sine  # 训练
```

## 与 controller_spec.md 的差异

- **符号约定**：spec 用 CW+（顺时针正），实现用 CCW+（逆时针正）。`lat_truck.py` Steps 4/5/7 的符号与 spec 相反，三处翻转自洽，最终输出数学等价
- **横向控制器**：spec §2.5 Steps 1-10 全部实现，无遗漏
- **纵向控制器**：spec §4.5 Steps 1-6 + IIR 滤波全部实现；Steps 7-9（GearControl / CalFinalTorque / 输出分配）有意跳过，直接输出加速度 (m/s²) 而非扭矩/刹车/档位
- **纵向时间坐标**：spec 用 `absolute_time`，实现用 `relative_time`（仿真中等价）；速度误差用 1.0s 预瞄点（更具前瞻性）

## 参考文档

- `docs/controller_spec.md` — 控制器完整规格
- `docs/tunable_params_analysis.md` — 可调参数分析
- `docs/plans/2026-03-02-differentiable-tuning-v2-design.md` — V2 设计

## 备注

- CPU only（单步计算量极小，GPU kernel 启动开销反而拖慢）
- 长序列 BPTT（>128 步）存在梯度爆炸风险，训练中用 `nan_to_num_` + `clip_grad_norm_` 缓解
