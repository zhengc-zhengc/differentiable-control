# 控制模块：信号流、控制结构与可微调参数分析

> 适用车型：重卡（ lat_controller_truck.cc + lon_controller.cc ）
>          电动牵引车（ lat_controller.cc + lon_controller.cc ）
> 配置文件： conf/control_conf_truck.pb.txt
> 运行频率：50 Hz（ts = 0.02 s）

---

## 一、控制器的整体信号流

```
┌─────────────────────────────────────────────────────────┐
│ 输入物理信号                                              │
│                                                          │
│ ① GlobalPose：车辆 ENU 位置(x,y)、航向角 yaw(度)          │
│ ② CCANInfoStruct：车速(km/h)、方向盘反馈角(度)、横摆角速率(rad/s) │
│            纵向加速度(m/s²)、档位反馈、制动踏板状态        │
│ ③ ADCTrajectory：规划路径点(位置、切线方向θ、曲率κ、速度v、加速度a) │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │  信号预处理    │
                  └──────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
  ┌────────────────────────┐  ┌──────────────┐
  │ 横向控制器（按车型选一）  │  │  纵向控制器    │
  │ 重卡: LatControllerTruck│─→│  (后执行)     │
  │ 电拖: LatController     │曲率│              │
  │ (先执行)                │far│              │
  └────────────────────────┘  └──────────────┘
              │                       │
              ▼                       ▼
   方向盘转角指令(°)      驱动扭矩(Nm) / 制动减速度(m/s²)
                                + 目标档位(N/D/P)
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
               ControlCommand 发布
            /rina/s2s/ADUControlMessage
```

---

## 二、信号预处理详解

### 2.1 公共预处理（两个控制器共用）

| 原始信号 | 处理内容 | 结果 |
|--------|--------|------|
| 车速 `ccan221_abs_vehspdlgt()` (km/h) | ÷ 3.6 | 车速 m/s |
| 航向角 `euler_angles().z()` (度) | × π/180 | 航向角 rad |
| 规划轨迹 | `TrajectoryAnalyzer` 封装：支持按位置查询、按时间查询、Frenet 变换 | 轨迹查询接口 |

### 2.2 横向控制器预处理（ `LatControllerTruck` ，重卡）

#### Step 1 — 查询三个预瞄点

```
当前点:    QueryNearestPointByPosition(x, y)           → target_point_currt
近预瞄点:  QueryNearestPointByRelativeTime(t + near_point_time_) → target_point_near
远预瞄点:  QueryNearestPointByRelativeTime(t + far_point_time_)  → target_point_far
```

`near_point_time_` 和 `far_point_time_` 来自**查找表**（按车速插值），单位秒。

#### Step 2 — 计算误差

```
lateral_error = cos(θ_ref)·Δy - sin(θ_ref)·Δx    // 有符号横向偏差，米
heading_error = NormalizeAngle(θ_vehicle - θ_currt) // 航向误差，rad
curvature_far_ = target_point_far.kappa()          // 远预瞄曲率（写入 cmd，供纵向读取）
```

#### Step 3 — 铰接修正（ `calculateRealTheta` ）

```
real_theta = heading_error - atan(kLh × yawrate / vehicle_speed)
// kLh = 0，当前未启用铰接补偿，保留接口
```

#### Step 4 — 横向偏差转换为等效角（ `calculateTargetTheta` ）

```
prev_dist    = max(vehicle_speed × prev_time_dist,  kMin_prev_dist=5.0)
error_angle = atan(lateral_error / prev_dist)
target_theta = clamp(error_angle, ±max_theta_deg)  // max_theta_deg 来自查找表
```

### 2.3 横向控制器预处理（ `LatController` ，电动牵引车）

#### Step 1 — 查询四个预瞄点

```
end_pnt_time_    = Lookup1D(|curvature_near|) × Lookup1D(speed_kph)  // 经 IIR 滤波
near_point_time = end_pnt_time_ × near_coff_in_end_pnt  (= 0.02)
far_point_time  = end_pnt_time_ × far_coff_in_end_pnt   (= 0.9)

当前点:    QueryNearestPointByPosition(x, y)
近预瞄点:  QueryNearestPointByRelativeTime(t + near_point_time)
远预瞄点:  QueryNearestPointByRelativeTime(t + far_point_time)
前馈点:    与当前点同时刻（取当前点曲率作前馈基准）
```

#### Step 2 — 计算横向偏差与航向误差

```
lateral_error = cos(θ_ref)·Δy - sin(θ_ref)·Δx    // 有符号横向偏差，米

heading_error_from_dy:
  dy2heading_time = Lookup1D(lateral_error) × Lookup1D(curvature_coff)
  raw = dy_weight(lateral_error) × heading_error_from_dy_gain
      × atan(lateral_error / max(dy2heading_dist_min=20, v × dy2heading_time))
  若 |lateral_error| > add_heading_err_from_dy_max_lat_err 阈值才生效
  经 IIR 滤波（filter_coff = 0.15）

heading_error_near_ = heading_weight × (θ_vehicle - θ_near) + heading_error_from_dy
heading_error_far_  = heading_weight × (θ_vehicle - θ_far)
                    + heading_error_from_dy × (near_coff / far_coff)
```

#### Step 3 — 速度窗口增益

```
spd_window_gain_ = 线性插值(speed, road_turn_spd=150, road_normal_lka_spd=200)
far_weight = far_weight_raw × spd_window_gain_    // 低速时压制远预瞄权重
```

### 2.4 纵向控制器预处理（ `LonController` ）

#### Step 1 — Frenet 坐标变换

```
QueryMatchedPathPoint(x, y)         → matched_point
ToTrajectoryFrame(x, y, θ, v, ...)  → s_matched, s_dot_matched, d_matched
```

#### Step 2 — 查询三个时间基准点

```
reference_point  = QueryNearestPointByAbsoluteTime(t_now)
preview_point    = QueryNearestPointByAbsoluteTime(t_now + preview_time)
                   preview_time = preview_window × ts = 5.0 × 0.02 = 0.1 s
preview_point_spd = QueryNearestPointByAbsoluteTime(t_now + preview_window_for_speed_pid × ts)
                  = t_now + 50 × 0.02 = t_now + 1.0 s
```

#### Step 3 — 误差计算

```
station_error       = s_ref - s_matched        // 路径弧长偏差，米
speed_error         = v_ref - v_matched        // 速度偏差，m/s
preview_speed_error = v_preview_ref - v_now    // 预观察点速度误差
preview_accel_ref   = a_preview                // 预观察点参考加速度
```

#### Step 4 — 站位误差保护

```
station_error_limited = clamp(station_error, ±station_error_limit=8.0)
// 低速时额外约束：防止静止时误差过大引起冲击
if speed < 1.0 km/h:
    station_error_fnl = min(0, station_error_limited)  // 仅允许追赶，不允许倒退
```

---

## 三、控制律详解

### 3.1 横向控制律（重卡，几何预瞄 + 曲率前馈）

```
[反馈路径]
  target_curvature = -[(target_theta - real_theta) + (target_dt_theta - real_dt_theta) × T_dt]
                     / max(reach_time_theta × v, kMin_reach_dis_theta=3.0)

  steer_fb = atan(target_curvature × wheelbase) × (180/π) × steer_ratio × slip_param
  steer_fb → rateLimitFilter(kRate_limit_fb = 120 °/s)

[前馈路径]
  steer_ff = atan(curvature_far × wheelbase) × (180/π) × steer_ratio × slip_param
  steer_ff → rateLimitFilter(kRate_limit_ff = 165 °/s)

[合并输出]
  steer_raw = clamp(steer_fb + steer_ff,  ±max_steer_angle)  // max_steer_angle 来自查找表
  steer_out = rateLimitFilter(steer_raw, kRate_limit_total = 300 °/s)
  → cmd.steering_target
```

**本控制器无 PID 积分器**，核心控制律是几何预瞄比例控制。

### 3.2 横向控制律（电动牵引车， `LatController` ）

核心算法为 **Matlab/Simulink 自动生成的双预瞄点 PID + 速度自适应前馈**，叠加三级滤波链。

```
[Matlab PID 反馈核心  ads_control__()]
  输入: heading_error_near_, heading_error_far_
       Kp = pid_p_param(speed),  Ki = Kp × pid_i_coff_base_p (=0.125)
       近/远路各自的 P/I 限幅、权重、积分重置参数
  输出: angD_DsrdStrAng（期望前轮转角，rad）

  首次使能 或 轨迹横向重规划 → 积分上下限归零（清空积分器）

[前馈路径]
  ff_curvature_ = curvature_far_  (use_far_point_as_ff=1 时)
  turn_ang_req_ff = ang_req_ff_coff × atan(L × κ_ff)
                  × (1 + v² / v_char²)    // 速度自适应增益，v_char = 120/3.6 m/s
                  × steer_ratio × (180/π) × lateral_control_use_ff

  ang_req_ff_coff = ang_req_ff_base_curv(κ_current) × ang_req_ff_base_spd(speed)

[反馈路径]
  turn_ang_req_fb = -(angD_DsrdStrAng × steer_ratio - steer_angle_req_add)

[合并 + 饱和限幅]
  steer_raw = turn_ang_req_ff + turn_ang_req_fb
  sat_max   = max(|atan(L × κ_ff) × steer_ratio| × angle_req_max_coff_base_ff,
                  angle_req_sat_min_lim = 18°)
  steer_sat = clamp(steer_raw, ±sat_max)

[输出滤波链]
  → rateLimitFilter(angle_req_rate_lim(en_time))  // 软使能速率渐进
  → LowPassFilter(τ = 0.01s)                      // 硬编码
  → LowPassFilter(τ = 0.03s)                      // 硬编码
  → bandstop_filter([1,52.51,2527]/[1,100.54,2527]) // 抑制机械共振，硬编码
  → cmd.steering_target
```

**与重卡版的关键区别**：有 PID 积分器（含积分重置机制）；前馈含速度自适应增益 (1 + v²/v_char²)；输出经三级滤波链而非速率限制。

### 3.3 纵向控制律（级联 PID + 扭矩模型）

```
[站位环 PID]
  speed_offset = station_pid.Control(station_error_fnl)
  // 站位误差 → 速度补偿量 (m/s)

[速度环 PID]（低速/高速自动切换）
  speed_input = speed_offset + preview_speed_error
  speed_input = clamp(speed_input, ±speed_controller_input_limit=5.0)
  acc_closeloop = speed_pid.Control(speed_input)
  // 速度误差 → 期望加速度 (m/s²)

[前馈叠加]
  acc_cmd = acc_closeloop + acc_cmd_use_preview_point_a × preview_accel_ref
          + enable_slope_offset × slope_compensation
  // 当前版本：grade_percent 硬编码 0，坡度补偿实际未生效

[加速度限幅 CalFinalAccCmd]
  acc_up_lim  = acc_up_lim_table(veh_spd)       // 查找表
  acc_low_lim = acc_low_lim_table(veh_spd)      // 查找表
  if curvature_far < -0.0075:                    // 急弯收紧
      acc_up_lim  ×= 0.75
      acc_low_lim ×= 0.60
  acc_clamped = clamp(acc_cmd, acc_low_lim, acc_up_lim)
  acc_rate_up  = acc_up_rate_table(acc_prev) × acc_rate_gain_table(veh_spd)
  acc_rate_dn  = acc_down_rate_table(acc_prev)  [低速时使用 acc_standstill_down_rate]
  acc_final = clamp(acc_clamped, acc_prev + acc_rate_dn, acc_prev + acc_rate_up)
  acc_out = LowPassFilter(acc_final, α=0.15)     // 一阶低通

[扭矩模型 CalFinalTorque]（仅 acc_out > 0 时执行）
  Road_Slope = 加速度计读数 - dv/dt × g⁻¹       // 坡度估算
  F_air     = 0.5 × Cd × ρ_air × A_wind × v²
  F_rolling = Crr × m × g × cos(slope)
  F_slope   = m × g × sin(slope)
  F_inertia = Cδ × m × acc_out
  F_resist  = F_air + F_rolling + F_slope + F_inertia
  Error_ax  = acc_out - a_measured
  F_PI      = Kp_ax × Error_ax  [+ 积分项，当前积分项实现有bug，Ki未真正累计]
  T_out     = (F_resist + F_PI) × r_tyre / (η × i_trans)
  T_out     = clamp(T_out, 0, torque_upper_limit)
  → cmd.target_torque

[制动输出]（acc_out ≤ 0）
  → cmd.brake = acc_out    // 直接以减速度形式输出

[档位状态机]
  N→D: acc > 0.01 && v ≤ 0.5 m/s，持续 50 拍(1s)
  D→N: acc < 0.001 && v ≤ 0.01 m/s，持续 50 拍(1s)
  N→P: N 档且无指令且 v ≈ 0，持续 200 拍(4s)
```

---

## 四、参数的三种类型与分类原则

控制模块的所有数值可以归为三类，**调参只针对第三类**：

| 类型 | 含义 | 例子 | 是否调参目标 |
|------|------|------|------------|
| 物理传感信号 | 实时采集的车辆状态 | 车速、位置、航向角、加速度计、方向盘反馈 | 否（输入） |
| 车辆物理常数 | 描述被控对象本身的固有属性 | 轮距、轴距、转向比、轮胎半径、传动效率、传动比 | 否（固定标定值） |
| 控制器设计参数 | 影响控制律行为的人为设定量 | PID 增益、预瞄时间、权重系数、限幅边界 | **是** |

第三类参数又可按结构细分为：

- **标量参数**：单一数值，直接出现在控制律公式里（如 `kp`、`ki`）
- **查找表参数**：分段线性函数的节点集合（如 `acc_up_lim_table`），每个节点的 value 都是一个标量参数，只是被组织成了随工况变化的形式
- **固定安全约束**：与控制性能无关、保障安全的边界（如速率限制硬常量 `kRate_limit_fb=120°/s`）

---

## 五、可微调参数完整清单

### 5.1 纵向控制器参数

#### PID 增益（最核心）

| 参数 | 配置路径 | 当前值 | 作用 |
|------|--------|-------|------|
| `station_pid.kp` | `lon_controller_conf.station_pid_conf.kp` | 0.25 | 站位误差→速度补偿的比例增益 |
| `station_pid.ki` | `...station_pid_conf.ki` | 0.0 | 站位积分（当前关闭） |
| `low_speed_pid.kp` | `...low_speed_pid_conf.kp` | 0.35 | 低速（≤ switch_speed m/s）速度 PID 比例增益 |
| `low_speed_pid.ki` | `...low_speed_pid_conf.ki` | 0.01 | 低速速度 PID 积分增益 |
| `high_speed_pid.kp` | `...high_speed_pid_conf.kp` | 0.34 | 高速速度 PID 比例增益 |
| `high_speed_pid.ki` | `...high_speed_pid_conf.ki` | 0.01 | 高速速度 PID 积分增益 |
| `switch_speed` | `lon_controller_conf.switch_speed` | 3.0 m/s | 低/高速 PID 切换点 |

#### 前馈参数

| 参数 | 配置路径 | 当前值 | 作用 |
|------|--------|-------|------|
| `acc_cmd_use_preview_point_a` | `lon_controller_conf` | 1.0 | 预观察加速度前馈权重 [0,1] |
| `a_preview_point_filt_coff` | `lon_controller_conf` | 0.05 | 预观察加速度 IIR 滤波系数 |
| `preview_window_for_speed_pid` | `lon_controller_conf` | 50 拍 (1.0s) | 速度前馈的预观察时间窗 |

#### 扭矩模型参数（ `control_gflags.cc` ）

| 参数 | GFlag 名称 | 当前值 | 作用 |
|------|-----------|-------|------|
| `accel_to_torque_kp` | `FLAGS_accel_to_torque_kp` | 1000 N/(m/s²) | 加速度跟踪 PI 的 P 增益 |
| `accel_to_torque_ki` | `FLAGS_accel_to_torque_ki` | 5 | 加速度跟踪 PI 的 I 增益 |
| `coef_rolling` | `FLAGS_coef_rolling` | 0.013 | 滚动阻力系数 Crr |
| `coef_cd` | `FLAGS_coef_cd` | 0.6 | 风阻系数 Cd |
| `coef_delta` | `FLAGS_coef_delta` | 1.05 | 转动惯量修正系数 Cδ |

> **注意**：Crr、Cd、Cδ 在物理上属于车辆参数，但在本代码中它们通过 GFlags 以可修改形式存在，且直接影响前馈扭矩的计算精度，**是扭矩前馈标定的调参目标**。

#### 加速度限幅查找表（每个节点的 value 均为可调参数）

| 表名 | 索引变量 | 节点数 | 当前值范围 | 作用 |
|------|--------|-------|----------|------|
| `acc_up_lim_table` | 车速 (km/h) | 5点 | 1.2～1.6 m/s² | 各速度段最大允许加速度 |
| `acc_low_lim_table` | 车速 (km/h) | 6点 | -0.1～-3.5 m/s² | 各速度段最大允许减速度 |
| `acc_up_rate_table` | 上一帧加速度 (m/s²) | 6点 | 0.035～0.045 m/s²/拍 | 加速度上升速率限制 |
| `acc_down_rate_table` | 上一帧加速度 (m/s²) | 9点 | -0.02～-0.03 m/s²/拍 | 加速度下降速率限制 |
| `acc_rate_gain_table` | 车速 (km/h) | 5点 | 1.0～1.5 | 速率限制随速度的缩放增益 |

### 5.2 横向控制器参数（重卡版 `LatControllerTruck` ）

横向控制器**无 PID 积分器**，所有参数都通过查找表形式存在。

#### 控制律核心参数（查找表节点值）

| 表名 | 配置字段 | 索引变量 | 作用 |
|------|--------|--------|------|
| `yawrate_gain_table` | 映射至 `data_max_theta_deg_` | 车速 (km/h) | 各速度下最大允许航向误差角（相当于横向 P 增益上限） |
| `theta_yawrate_gain_table` | 映射至 `data_prev_time_dist_` | 车速 (km/h) | 预瞄距离时间系数（控制反馈收敛的"弹簧刚度"） |
| `theta_yawrate_gain_table2` | 映射至 `data_reach_time_theta_` | 方向盘角度请求 (°) | 收敛时间因子（影响横向误差收敛速度） |
| `end_pnt_time_table` | 映射至 `data_prew_time_dt_theta_` | 车速 (km/h) | 角速度误差的预瞄时间 T_dt（影响 dθ/dt 项的权重） |
| `dy2heading_time_table` | 映射至 `data_near_point_time_` | 车速 (km/h) | 近预瞄点时间（影响横向偏差到目标角的转换距离） |
| `dy2_heading_time_coff_table` | 映射至 `data_far_point_time_` | 曲率 | 远预瞄点时间修正系数 |
| `angle_req_max_vlu_table` | 映射至 `data_max_steer_angle_` | 曲率 | 各曲率下方向盘转角的幅值限制 |
| `pid_p_param_table` | 映射至 `data_slip_param_` | 车速 (km/h) | 侧滑修正增益 slip_param（在转向角到曲率的转换中使用） |

#### 硬编码速率限制（不在配置文件中，代码级常量）

| 常量 | 位置 | 当前值 | 说明 |
|------|------|-------|------|
| `kRate_limit_fb` | `lat_controller_truck.cc:13` | 120 °/s | 反馈通道速率限制 |
| `kRate_limit_ff` | `lat_controller_truck.cc:14` | 165 °/s | 前馈通道速率限制 |
| `kRate_limit_total` | `lat_controller_truck.cc:15` | 300 °/s | 合并后总速率限制 |
| `kMin_prev_dist` | `lat_controller_truck.cc:16` | 5.0 m | 预瞄距离最小值 |
| `kMin_reach_dis_theta` | `lat_controller_truck.cc:17` | 3.0 m | 收敛距离最小值 |

### 5.3 横向控制器参数（电动牵引车版 `LatController` ）

#### PID 增益（含积分器，核心调参目标）

| 参数 | 配置路径 | 当前值 | 作用 |
|------|--------|-------|------|
| `pid_p_param_table` (各节点 value) | `lat_controller_conf.pid_p_param_table` | 全 1.0 | f(速度 kph) → Kp 基数 |
| `pid_i_coff_base_p` | `lat_controller_conf` | 0.125 | Ki = Kp × 此系数 |
| `near_weight_in_angle_req` | `lat_controller_conf` | 1.45 | 近预瞄 PID 输出权重 |
| `far_weight_in_angle_req` | `lat_controller_conf` | 0.35 | 远预瞄 PID 输出权重基数（还受速度窗口增益调制） |
| `near_pid_i_max` / `near_pid_i_min` | `lat_controller_conf` | ±3.0 | 近预瞄积分器限幅 |
| `far_pid_i_max` / `far_pid_i_min` | `lat_controller_conf` | ±0.001 | 远预瞄积分器限幅 |
| `near_pid_p_max` / `near_pid_p_min` | `lat_controller_conf` | ±20 | 近预瞄比例项限幅 |
| `far_pid_p_max` / `far_pid_p_min` | `lat_controller_conf` | ±0.001 | 远预瞄比例项限幅 |
| `near_pid_i_reset_rate` | `lat_controller_conf` | 0.005 | 近预瞄积分重置速率（× 速度窗口增益） |
| `far_pid_i_reset_rate` | `lat_controller_conf` | 0.005 | 远预瞄积分重置速率 |

#### 预瞄时间参数（查找表）

| 表名 | 索引变量 | 当前值 | 作用 |
|------|--------|-------|------|
| `end_pnt_time_table` | \|curvature_near\| | 0～0.3 s | 预瞄时间基数（按近端曲率查） |
| `end_pnt_time_mody_base_spd_table` | 速度 (kph) | 0.75～1.0 | 预瞄时间速度修正系数 |
| `near_coff_in_end_pnt` | — | 0.02 | 近预瞄时间 = end_pnt_time × 此值 |
| `far_coff_in_end_pnt` | — | 0.9 | 远预瞄时间 = end_pnt_time × 此值 |

#### 前馈参数

| 表名 / 参数 | 索引变量 | 当前值 | 作用 |
|------------|--------|-------|------|
| `ang_req_ff_base_curv_table` | 当前曲率 | 全 1.0 | 前馈系数的曲率分量 |
| `ang_req_ff_base_spd_table` | 速度 (kph) | 0.88～1.0 | 前馈系数的速度分量（低速减小前馈） |
| `lateral_control_use_ff` | — | 1 | 前馈总开关（0=纯反馈） |
| `angle_req_max_coff_base_ff` | — | 1.5 | 饱和限幅 = ff 基准角 × 此系数 |
| `angle_req_sat_min_lim` | — | 18° | 饱和限幅下限（防止小曲率时限幅过小） |

#### 横向偏差→航向误差转换参数

| 表名 / 参数 | 索引变量 | 当前值 | 作用 |
|------------|--------|-------|------|
| `dy_weight_table` | lateral_error (m) | 全 2.0 | lateral_error 转换权重 |
| `heading_weight_table` | lateral_error (m) | 全 1.0 | heading_error 混合权重 |
| `dy2heading_time_table` | lateral_error (m) | 全 0.1 s | dy→heading 虚拟预瞄时间 |
| `heading_error_from_dy_gain` | — | 1.0 | dy→heading 转换总增益 |
| `heading_error_from_dy_filter_coff` | — | 0.15 | dy→heading 转换项的 IIR 系数 |
| `dy2_heading_time_coff_table` | 曲率 | 全 1.0 | dy2heading 时间的曲率修正系数 |

#### 速度窗口参数（调制远预瞄权重）

| 参数 | 当前值 | 作用 |
|------|-------|------|
| `road_turn_spd_vlu` | 150 km/h | 低于此速度时远预瞄权重降至 0 |
| `road_normal_lka_spd_vlu` | 200 km/h | 高于此速度时远预瞄权重恢复全值 |

#### 硬编码滤波器参数（代码级，不在配置文件）

| 参数 | 位置 | 当前值 | 说明 |
|------|------|-------|------|
| LowPassFilter 1 | `lat_controller.cc:77` | τ = 0.01 s | 第一级低通，平滑高频 |
| LowPassFilter 2 | `lat_controller.cc:81` | τ = 0.03 s | 第二级低通，进一步平滑 |
| bandstop filter | `lat_controller.cc:84` | num=[1,52.51,2527] den=[1,100.54,2527] | 陷波，抑制机械共振 |

---

## 六、查找表与控制器参数的关系

### 6.1 本质是同一回事

查找表的本质是：**将一个标量控制参数扩展为随工况变化的分段线性函数。**

```
标量参数:     kp = 0.35
              ⇕ 等价于
查找表参数: kp(v) = Lookup1D(v; [(0, 0.35), (10, 0.34), (20, 0.30), ...])
```

两者在控制律中的地位完全相同——都是人为设定的、影响控制器行为的设计量。查找表只是额外引入了"增益随工况调度（Gain Scheduling）"的机制，使参数可以适应不同车速段的动力学特性。

因此：**查找表的每一个节点 (index, value) 中，value 就是该工况点的控制参数**，所有 value 都是可微调参数。

### 6.2 三类不同性质的查找表

| 类别 | 例子 | 性质 | 可微调策略 |
|------|------|------|----------|
| 增益调度表（控制参数） | `acc_up_lim_table`、`pid_p_param_table` | 描述控制器在不同工况下的增益 | **是调参目标**，每个节点 value 均可调 |
| 底盘标定表（物理映射） | `calibration_table(speed, acc → cmd)` | 描述底盘驱动系统的输入/输出非线性关系，由实车标定获得 | **不调**，调它等于同时改控制器和被控对象模型 |
| 安全保护表（约束包络） | `angle_req_rate_lim_table`（软使能速率渐进） | 工程安全约束，与控制性能无关 | **不纳入梯度优化**，保留为硬约束 |

### 6.3 可微调参数化时的建议

对增益调度表进行可微优化时，有两种方案：

**方案 A：直接优化节点值**

- 将每个节点的 value 作为独立参数
- 优点：灵活，可表达任意形状
- 缺点：参数量多，需正则化约束单调性或平滑性

**方案 B：用参数化函数替换查找表**

- 例如用 `f(v) = a × exp(-b × v) + c` 替换多点查找表
- 优点：参数少，物理含义明确，天然平滑
- 缺点：表达能力受函数形式限制

对于本模块，**推荐从方案 A 入手**（直接优化节点值），因为现有查找表节点已是精心选取的工况断点，可以保留索引轴不变，只优化 value 轴。

---

## 七、参数优先级与调参建议

```
【第一层：直接影响跟踪误差的反馈增益】（最先优化）
  纵向 PID: station_kp, low/high_speed kp, ki
  横向增益表: theta_yawrate_gain_table（决定收敛速度）

【第二层：影响系统鲁棒性和平滑性的前馈与滤波参数】
  acc_cmd_use_preview_point_a（前馈强度）
  a_preview_point_filt_coff（前馈平滑度）
  pid_p_param_table (slip_param，影响横向前馈精度)

【第三层：安全包络，影响极限工况下的表现】
  acc_up_lim_table, acc_low_lim_table（加速度幅值上下限）
  acc_up_rate_table, acc_down_rate_table（加速度平滑度）
  angle_req_max_vlu_table（转向幅值限制）

【固定不动：底盘标定与安全约束】
  calibration_table（底盘标定，独立标定获取）
  kRate_limit_fb/ff/total（硬编码速率限制，改动需修改代码）
  acc_standstill_down_rate（静止特殊保护逻辑）
```
