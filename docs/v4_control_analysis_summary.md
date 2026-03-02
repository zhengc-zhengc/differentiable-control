# Control 模块横纵向控制器深度分析 — 文档总结

> 来源: v4 录屏 (`屏幕录制 2026-02-27 104557.mp4`) + 10 张手动截图
> 总结日期: 2026-02-27

## 文件范围

| 文件 | 路径 |
|------|------|
| 横向控制器实现 | `controller/lat_controller_truck.cc` |
| 横向控制器头文件 | `controller/lat_controller_truck.h` |
| 纵向控制器实现 | `controller/lon_controller.cc` |
| 纵向控制器头文件 | `controller/lon_controller.h` |
| 控制器编排器 | `controller/controller_agent.cc` |
| 主控组件 | `onboard/control_component.cc` |

---

## 一、整体概览

> 注：文档中第一节（整体概览）在手动截图中未包含，以下信息来自录屏帧分析。

`ControlComponent`（`onboard/control_component.cc`）以 **50Hz** 由 `TimerComponent` 驱动，每个周期固定三个工序：

1. 读取最新的**定位/底盘/CAN**，更新 `LocalView`
2. 用 `ControllerAgent` 进入**控制循环**（先横向后纵向依次执行）
3. 把 `ControlCommand` 写到 **CAN**

横纵向控制器共享同一 `ADCTrajectory` 对象，各自独立查询所需的轨迹点。

---

## 二、横向控制器（LatControllerTruck）

### 2.1 输入

| 物理量 | 来源/数据结构 | C++ 字段 |
|--------|-------------|----------|
| 车辆 ENU 坐标 | GlobalPose | `position_enu().x/y()` |
| 航向角（偏航）| GlobalPose | `euler_angles().z()`，单位度，内部转弧度 |
| 车速 | CCANInfoStruct（底盘 CAN）| `ccan221_abs_vehspdlgt()`，单位 km/h |
| 方向盘反馈角 | CCANInfoStruct | `ccan175_eps1_eps1_steerwheelangle()`，单位度 |
| 横摆角速率 | CCANInfoStruct | `ccan242_esp_yawrate()`，单位 rad/s |
| 规划轨迹 | ADCTrajectory | 路径点数组（位置、航向、曲率、速度、时间戳）|

控制器从轨迹中提取**三个预瞄点**，都通过 `TrajectoryAnalyzer` 查询：

- **当前点**：按位置最近匹配
- **近预瞄点**：当前时刻 + `near_point_time`（速度相关，约 0.1s）
- **远预瞄点**：当前时刻 + `far_point_time`（速度相关，约 1.0s）

### 2.2 输出

| 物理量 | 目标数据结构 | C++ 字段 |
|--------|-------------|----------|
| 方向盘转角指令 | ControlCommand | `set_steering_target()`，单位度 |
| 当前点路径曲率 | ControlCommand | `set_path_curvature_current()` |
| 近预瞄点曲率 | ControlCommand | `set_path_curvature_near()` |
| 远预瞄点曲率 | ControlCommand | `set_path_curvature_far()`（纵向控制器读取此值）|

### 2.3 控制算法概述

算法类型：**多点预瞄 + 曲率前馈**，无经典 PID。

核心思路：

- 用**近预瞄点**的横向偏差和航向误差计算**反馈转角**
- 用**远预瞄点**的路径曲率计算**前馈转角**（弯道预补偿）
- 两路叠加后经**三级限速**（反馈速率 ≤120°/s、前馈速率 ≤165°/s、合计速率 ≤300°/s）输出

### 2.4 控制流程（带函数名）

```
ComputeControlCommand()                    // lat_controller_truck.cc 主入口
│
├─ 读入传感器数据
│   速度 kph→mps、方向盘反馈角、横摆角速率、航向角
│
├─ UpdateState()
│   └─ ComputeLateralErrors()
│       查询三个预瞄点，计算：
│       heading_error  = 车头方向 − 近预瞄点切线方向
│       lateral_error  = 车辆到近预瞄点路径的垂直偏差（米）
│       curvature_near_= 近预瞄点曲率
│       curvature_far_ = 远预瞄点曲率（同时写入 cmd）
│
├─ 按当前速度查速度相关查找表
│   预瞄时间、最大横向误差角、速率限制等参数均随车速变化
│
├─ calculateRealTheta()
│   修正铰接点影响：
│   real_theta = heading_error − atan(轴距_铰接 × 横摆角速率 / 车速)
│
├─ calculateRealDtTheta()
│   计算"实际横摆角速率超出路径曲率引起部分"：
│   real_dt_theta = −(横摆角速率 − curvature_far × 车速)
│
├─ calculateTargetTheta()
│   把横向偏差转换为目标航向误差角：
│   prev_dist = max(车速 × 预瞄时间, 最小预瞄距离)
│   error_angle = atan(lateral_error / prev_dist)
│   target_theta = 截幅后的 error_angle（幅值 ≤ max_theta_deg）
│
├─ calculateTargetCurvature()
│   反馈控制律：
│   target_curv = −[(target_theta − real_theta)
│                  + (target_dt_theta − real_dt_theta) × 预瞄时间]
│                  / max(收敛时间 × 车速, 最小距离)
│
├─ [反馈路径]
│   calculateSteeringAngle(target_curv)    // 曲率→前轴转角→方向盘角度（含转向比）
│   rateLimitFilter(speed=120°/s)          // 反馈速率限制
│
├─ [前馈路径]
│   calculateSteeringAngle(curvature_far)  // 远预瞄曲率→方向盘角度
│   rateLimitFilter(speed=165°/s)          // 前馈速率限制
│
├─ 合并
│   target_steering_raw = clamp(反馈角 + 前馈角, ±max_steer_angle)
│   rateLimitFilter(speed=300°/s)          // 总速率限制
│
└─ cmd->set_steering_target(target_steering_angle_deg)
```

---

## 三、纵向控制器（LonController）

### 3.1 输入

| 物理量 | 来源/数据结构 | C++ 字段 |
|--------|-------------|----------|
| 车辆 ENU 坐标 | GlobalPose | `position_enu().x/y()` |
| 航向角 | GlobalPose | `euler_angles().z()`，度→弧度 |
| 车速 | CCANInfoStruct | `ccan221_abs_vehspdlgt()`，km/h |
| 车速有效性 | CCANInfoStruct | `ccan221_abs_vehspdlgtstatus()`，0=有效 |
| 纵向加速度（加速度计）| CCANInfoStruct | `ccan242_esp_algt()`，m/s² |
| 当前档位反馈 | CCANInfoStruct | `ccan123_vcu_displaygear()`，档位编码 |
| 制动踏板状态 | CCANInfoStruct | `ccan267_ehb_brakeactive()` |
| 远预瞄路径曲率 | ControlCommand（横向控制器已写入）| `path_curvature_far()` |
| 规划轨迹 | ADCTrajectory | 路径点数组（弧长 s、参考速度 v、加速度 a）|

### 3.2 输出

| 物理量 | 目标数据结构 | C++ 字段 |
|--------|-------------|----------|
| 驱动扭矩指令 | ControlCommand | `set_target_torque()`，单位 Nm，仅加速时输出 |
| 制动减速度指令 | ControlCommand | `set_brake()`，单位 m/s²（负值），仅减速时输出 |
| 目标档位 | ControlCommand | `set_gear_location()`，枚举：N/D/P/R |

### 3.3 控制算法概述

算法类型：**级联 PID + 车辆动力学前馈 + 坡度补偿 + 档位状态机**。

核心思路：

- 站位 PID 把路径弧长误差转为**速度补偿量**
- 速度 PID 把速度误差转为**期望加速度**
- 叠加**坡度重力补偿**和**预观察加速度前馈**
- 经幅值 + 速率**双重限制**后进入扭矩模型
- 加速度 > 0 → 扭矩模型输出驱动扭矩；≤0 → 直接输出制动减速度

### 3.4 控制流程（带函数名）

```
ComputeControlCommand()                    // lon_controller.cc 主入口
│
├─ 更新 TrajectoryAnalyzer（若轨迹有更新）
│
├─ ComputeLongitudinalErrors()
│   └─ Frenet 坐标变换：
│       QueryMatchedPathPoint()       // 按位置查找最近轨迹点
│       ToTrajectoryFrame()           // 车辆状态→Frenet(s, ṡ, d, ḋ)
│       QueryNearestPointByAbsoluteTime() // 按时间查参考点、预瞄点
│   输出：
│       station_error        = 参考弧长 s_ref − 当前弧长 s
│       speed_error          = 参考速度 v_ref − 当前纵向速度 ṡ
│       accel_error          = 参考加速度 a_ref − 当前纵向加速度（曲率修正后）
│       preview_speed_error / preview_accel_ref（预观察点的对应量）
│
├─ 站位误差截幅
│   clamp(station_error, ±station_error_limit)
│   低速时额外保护，防止误差过大导致起步冲击
│
├─ 站位 PID
│   speed_offset = station_pid_controller_.Control(station_error, ts)
│   station_error → PID → 速度修正量（m/s）
│
├─ 速度 PID 输入合成
│   speed_controller_input = speed_offset + preview_speed_error
│   clamp(±speed_controller_input_limit)
│   低速/高速分别使用不同 PID 参数（切换点约 2.5 m/s）
│
├─ 速度 PID
│   acceleration_cmd_closeloop = speed_pid_controller_.Control(input, ts)
│   速度误差 → PID → 期望加速度（m/s²）
│
├─ 坡度补偿（当前版本 grade_percent 硬编码为 0，预留接口）
│   slope_offset = g × sin(atan(grade / 100))
│   经低通滤波后叠加
│
├─ 最终加速度合成
│   acceleration_cmd = PID输出 + 预观察加速度前馈 + 坡度补偿
│
├─ CalFinalAccCmd()                  // 幅值 + 速率双重限制
│   按当前速度查表获得：
│     acc_up_lim / acc_low_lim       // 加速度幅值上下限
│     acc_up_rate / acc_down_rate    // 加速度变化速率限制
│   若远端曲率 < −0.0075（急弯）：
│     acc_up_lim × 0.75, acc_low_lim × 0.6  // 弯道收紧限制
│   先幅值截幅，再速率截幅：
│     final_acc = clamp(limited_acc,
│                       pre + acc_down_rate,
│                       pre + acc_up_rate)
│   经一阶低通滤波（α≈0.15）输出
│
├─ GearControl()                     // 档位状态机（见 3.4.1）
│
├─ CalFinalTorque()                  // 驱动扭矩计算（仅 acceleration > 0 时调用，见 3.4.2）
│
└─ 分支输出
    若 acceleration > 0: cmd->set_target_torque(torque)
    若 acceleration ≤ 0: cmd->set_brake(acceleration)  // 制动减速度（负值）
    cmd->set_gear_location(gear_req)
```

### 3.4.1 档位状态机（GearControl）

当前档位（来自 CAN 反馈）→ 统一映射为 N / D / R 枚举

**N → D:**
- 条件：期望加速度 > 0.01 && 使能 && 车速 ≤ 0.5 m/s && 速度有效
- 计数器持续满足 50 拍（1秒）后，若实时车速 ≤ 0.25 m/s，请求切 D

**D → N:**
- 条件：期望加速度 < 0.001 && 车速 ≤ 0.01 m/s && 使能
- 计数器持续满足 50 拍（1秒）后，请求切 N

**N → P:**
- 条件：当前为 N && 使能 && 期望加速度 < 0.001 && 车速 ≤ 0.01 m/s && 速度有效
- 计数器持续满足 200 拍（4秒）后，请求切 P
- 若制动踏板激活则重置计数器

### 3.4.2 扭矩模型（CalFinalTorque）

```
CalFinalTorque(acceleration_cmd)

  估算道路坡度：
    维护近 10 帧（200ms）速度环形缓冲区
    dvdt = 速度差分估算实际加速度
    Road_Slope = (加速度计读数 − dvdt) / g

  计算各阻力分量：
    F_air     = 0.5 × Cd × 空气密度 × 迎风面积 × 车速²
    F_rolling  = Crr × 质量 × g × cos(坡角)
    F_slope    = 质量 × g × sin(坡角)
    F_inertia = C6 × 质量 × acceleration_cmd     // C6 含转动惯量修正

  PI 控制器修正加速度跟踪误差：
    Error_ax = acceleration_cmd − 实测加速度
    Force_PI = Kp × Error_ax + Ki × ∫Error_ax dt

  总驱动力：
    F_total = F_air + F_rolling + F_slope + F_inertia + Force_PI

  转换为输出扭矩：
    T = F_total × 轮胎半径 / (传动效率 × 传动比)
    D 档传动比 ≈ 14.02, R 档 ≈ 39.8
    结果截幅至配置上下限
```

---

## 四、横纵向联动

```
横向控制器                              纵向控制器
     │                                      │
     │  cmd->set_path_curvature_far()        │
     │ ───────────────────────────────→       │
     │  （远预瞄点路径曲率）                    │
     │                                      │
     │               CalFinalAccCmd() 中：    │
     │               若曲率 < −0.0075（急弯） │
     │               则收紧加速度幅值限制       │
```

- 纵向控制器**不**反馈任何数据给横向控制器，信息流**单向**
- 两者共享同一 `ADCTrajectory` 对象，独立查询各自所需的轨迹点

---

## 五、关键配置参数（运行时从 `control_conf_truck.pb.txt` 加载）

### 横向相关

| 参数 | 含义 |
|------|------|
| `near_point_time_table` | 速度相关的近预瞄时间（s）|
| `far_point_time_table` | 速度相关的远预瞄时间（s）|
| `max_theta_deg_table` | 速度相关的最大允许航向误差（度）|
| `pid_p_param_table` | 速度相关的侧滑修正系数 |

### 纵向相关

| 参数 | 含义 |
|------|------|
| `station_pid_conf` | 站位 PID（kp/ki/kd/kaw）|
| `low_speed_pid_conf` / `high_speed_pid_conf` | 速度 PID（低速/高速段）|
| `switch_speed` | 低/高速 PID 切换点（约 2.5 m/s）|
| `acc_up_lim_table` / `acc_low_lim_table` | 速度相关的加速度幅值上下限 |
| `acc_up_rate_table` / `acc_down_rate_table` | 加速度变化速率限制 |
| `FLAGS_coef_cd` / `rolling` / `delta` | 扭矩模型：空气阻力/滚阻/转动惯量系数 |
| `FLAGS_accel_to_torque_kp` / `ki` | 扭矩 PI 控制器增益 |

---

## 与现有代码骨架的差异分析

### 横向控制器（LatControllerTruck）

**v4 文档新发现的方法**（可补充到代码骨架）：

| 方法 | 描述 | 与 v3 骨架对应 |
|------|------|---------------|
| `UpdateState()` | 状态更新入口，调用 ComputeLateralErrors | 新发现 |
| `ComputeLateralErrors()` | 计算 heading_error, lateral_error, curvature_near/far | 新发现 |
| `calculateRealTheta()` | 修正铰接点影响 | 可能对应 v3 的 `calculateAbsHeadTheta()` |
| `calculateRealDtTheta()` | 计算横摆角速率超出曲率部分 | 可能对应 v3 的 `calculateMdlRTheta()` |
| `calculateSteeringAngle()` | 曲率→前轴转角→方向盘角度 | 可能对应 v3 的 `calculateSteerAngleByApply()` |
| `rateLimitFilter()` | 速率限制滤波器 | 可能对应 v3 的 `rateSetLimitDir()` |

**v4 确认的输出字段**（纠正旧版）：

| 字段 | 说明 |
|------|------|
| `set_steering_target()` | 方向盘转角指令，单位度（非弧/0.1°）|
| `set_path_curvature_current()` | 当前点路径曲率（新发现）|
| `set_path_curvature_near()` | 近预瞄点曲率（新发现）|
| `set_path_curvature_far()` | 远预瞄点曲率（新发现，纵向控制器读取此值）|

### 纵向控制器（LonController）

**v4 文档新发现的方法**：

| 方法 | 描述 |
|------|------|
| `ComputeLongitudinalErrors()` | Frenet 坐标变换 + 误差计算（QueryMatchedPathPoint, ToTrajectoryFrame, QueryNearestPointByAbsoluteTime）|
| `CalFinalAccCmd()` | 加速度幅值 + 速率双重限制，含弯道收紧逻辑 |
| `GearControl()` | 档位状态机（N/D/P/R 切换）|
| `CalFinalTorque()` | 基于车辆动力学的扭矩模型（含坡度估算、阻力计算、PI 修正）|

**v4 纠正的输出字段**（旧版有误）：

| 旧版（错误）| 实际（v4 文档）|
|-------------|--------------|
| `set_target_torque_MVG()` | `set_target_torque()`，单位 Nm |
| `set_front_brakePress_MRG()` | `set_brake()`，单位 m/s²（负值）|
| — | `set_gear_location()`，枚举 N/D/P/R |

### 配置参数纠正

| 旧版（错误）| 实际（v4 文档）|
|-------------|--------------|
| `near_point_line_table` | `near_point_time_table` |
| `far_point_line_table` | `far_point_time_table` |
| `max_error_log_table` | `max_theta_deg_table` |
| `arc_line_cte_table` | `acc_up_lim_table` / `acc_low_lim_table` |
| `PLAGE_avoid_to_torque_table3` | `FLAGS_accel_to_torque_kp/ki` |
