# V1 可微控制 Pipeline 设计

## 目标

用 Python/PyTorch 复现 LatControllerTruck + LonController（简化版），接入运动学自行车模型，
在常规轨迹（直线、圆弧、正弦、组合）上跑通 50Hz 闭环仿真，可视化跟踪效果。
**本版不做梯度优化**，仅验证控制器实现正确性。

## 文件结构

```
sim/
├── common.py              # lookup1d, rate_limit, PID, normalize_angle, TrajectoryPoint
├── trajectory.py           # 轨迹生成 + TrajectoryAnalyzer
├── vehicle.py              # 运动学自行车模型
├── controller/
│   ├── __init__.py
│   ├── lat_truck.py        # LatControllerTruck (spec §2.5, 10步)
│   └── lon.py              # LonController 简化版 (spec §4.5, Steps 1-6)
├── sim_loop.py             # 闭环仿真主循环
└── run_demo.py             # 入口: 轨迹生成 → 仿真 → matplotlib 可视化
```

## 模块设计

### common.py
- `lookup1d(table, x)` — 分段线性插值，边界 clamp
- `rate_limit(prev, target, rate, dt)` — 速率限制器
- `PID(kp, ki, kd, i_sat_min, i_sat_max)` — 带积分饱和
- `normalize_angle(angle)` — 归一化到 [-π, π]
- `TrajectoryPoint` — dataclass: x, y, theta, kappa, v, a, t

### trajectory.py
- 轨迹生成: straight_line, circle_arc, sine_curve, combined
- `TrajectoryAnalyzer`: position_query (最近点), time_query (绝对时间), frenet_transform

### vehicle.py — 运动学自行车
- 状态: [x, y, yaw, v]
- 输入: 前轮转角 delta + 加速度 acc
- 更新: x += v*cos(yaw)*dt, y += v*sin(yaw)*dt, yaw += v*tan(delta)/L*dt, v += acc*dt
- 参数: wheelbase L, dt=0.02s

### controller/lat_truck.py
- 完全按 controller_spec.md §2.5 实现
- 8 张查找表 T1-T8，用 spec 数值
- 内部状态: steer_fb_prev, steer_ff_prev, steer_total_prev
- 输入: 车辆状态 + TrajectoryAnalyzer
- 输出: 方向盘角度(°) + curvature_far

### controller/lon.py（简化版）
- 实现 Steps 1-6: Frenet 误差 → 站位 PID → 速度 PID → CalFinalAccCmd
- 跳过 Steps 7-9 (GearControl, CalFinalTorque)
- 直接输出加速度(m/s²)
- 5 张查找表 L1-L5
- 修正 road_slope 公式 bug

### sim_loop.py
- 50Hz 循环: lat → lon → 车辆更新 → 记录历史
- 方向盘角 / steer_ratio → 前轮转角

### run_demo.py
- 4 场景: 直线(10m/s), 圆弧(R=30m), 正弦, 组合(直线→弯→直线)
- 每场景 4 图: 轨迹对比, 横向误差, 速度跟踪, 转向角

## 关键参数
- wheelbase = 3.5m (重卡假设值)
- steer_gear_ratio = 17.5
- dt = 0.02s (50Hz)
- kMphToMps = 1/3.6
