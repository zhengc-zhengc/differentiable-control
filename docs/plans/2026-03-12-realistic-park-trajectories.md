# 实际园区工况参考轨迹设计

日期：2026-03-12

## 背景

当前训练和验证轨迹（圆弧、正弦、换道、组合）与园区低速重卡的实际工况脱节：
- 圆弧 R=30m 接近满舵极限，实际极少持续满舵行驶
- 曲率瞬变（0→1/R），实际道路有缓和曲线过渡
- 速度恒定，实际弯道前减速、弯后加速
- 缺少掉头、停靠起步等园区常见场景

## 车辆参数基准（hybrid_dynamic 模型）

| 参数 | 值 | 来源 |
|------|-----|------|
| 轴距 L | 2.8m (lf=1.354 + lr=1.446) | default.yaml |
| 转向比 | 16.39 | default.yaml |
| T7 低速最大方向盘角 | 110° | default.yaml |
| 最大前轮角 | 6.7° | 110°/16.39 |
| 最小转弯半径 | ~24m | L/tan(6.7°) |
| 质量 | 2440 kg | default.yaml |

## 新增轨迹类型

### 1. `generate_clothoid_turn(radius, turn_angle, speed, clothoid_ratio, lead_in, lead_out)`

直道 → clothoid（κ: 0→1/R） → 圆弧 → clothoid（κ: 1/R→0） → 直道

- clothoid 段长度 = `clothoid_ratio × 圆弧弧长`（默认 0.3）
- clothoid 内曲率线性变化：`κ(s) = (s / L_cl) × κ_max`
- `turn_angle > 0` 左转，`< 0` 右转
- 前后各有 `lead_in/lead_out` 直线段

### 2. `generate_uturn(radius, speed)`

`clothoid_turn(radius, π, speed)` 的 180° 特例。

### 3. `generate_stop_and_go(cruise_speed, stop_distance, accel_rate, decel_rate)`

直线上加减速停靠：
- 巡航段 → 匀减速到 0 → 停留 2s → 匀加速恢复巡航 → 巡航段
- 位置为直线，速度为梯形

### 4. `generate_park_route(speed)`

综合园区路线，拼接：
- 100m 直道 → 右转 90°(R=40m, clothoid) → 60m 直道 → 换道 3.5m → 80m 直道 → 左转 90°(R=35m, clothoid) → 50m 直道 → 减速停车

速度在弯前减速、弯后加速。

## 仿真方案

- 模型：`hybrid_dynamic`（MLP + Euler base）
- 参数：`default.yaml`（原始 C++ 参数，不训练）
- 模式：`differentiable=False`（V1 路径）

### 轨迹列表（15 条）

| # | 轨迹 | 速度 (km/h) |
|---|------|-------------|
| 1 | clothoid 右转 90° R=40m | 10 |
| 2 | clothoid 右转 90° R=40m | 15 |
| 3 | clothoid 左转 90° R=35m | 10 |
| 4 | clothoid 左转 90° R=35m | 15 |
| 5 | clothoid 弯道 R=80m 45° | 15 |
| 6 | clothoid 弯道 R=80m 45° | 25 |
| 7 | 掉头 R=30m | 5 |
| 8 | 掉头 R=30m | 8 |
| 9 | 换道 3.5m | 15 |
| 10 | 换道 3.5m | 25 |
| 11 | 停靠起步 | 15 |
| 12 | 停靠起步 | 25 |
| 13 | 综合园区路线 | 15 |
| 14 | 直道巡航 | 20 |
| 15 | 直道巡航 | 30 |

### 输出图表（每条轨迹）

1. 轨迹跟踪（x-y 平面，参考 vs 实际）
2. 横向误差 vs 时间
3. 航向误差 vs 时间
4. 速度跟踪（参考 vs 实际）
5. 转向角 vs 时间
6. 加速度指令 vs 时间

保存到 `sim/results/realistic_scenarios/`。
