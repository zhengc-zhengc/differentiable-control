/**
 * @file lon_controller.cc
 * @brief 纵向控制器实现文件
 *
 * 本文件实现了辅助驾驶内的纵向控制，包括：
 *  - 油门控制算法
 *  - 加速度计算与限制
 *  - 制动计算
 *  - 控制输出安全保障
 */

#include "controller/lon_controller.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>

// ?? 可能还有若干 project includes（base/logging.h、common/control_flags.h、common/log.h 或 logger.h、gnss/time/time.h 等），但由于截图 1 行号与可见内容无法精确对齐，具体包含列表待第 0 号/后续 agent 核对 [1.png]

namespace mlp {
namespace control {

namespace {

constexpr double kGravitationalAcc = 9.8;  ///< 重力加速度(m/s^2)

constexpr double kMps2Kph = 3.6;
constexpr double kKph2Mps = 1.0 / 3.6;
constexpr double kDeg2Rad = M_PI / 180.0;

// 门限相关参数
// ?? 以下 N2D/D2N/N2P 三个阈值的具体数值来自截图 1 的低分辨率区域，目视推测；待核对 [1.png]
constexpr int kCounterThresholdN2D = 20;  ///< N到D档计数器阈值
constexpr int kCounterThresholdD2N = 20;  ///< D到N档计数器阈值
constexpr int kCounterThresholdN2P = 50;  ///< N到P档计数器阈值
constexpr int kCounterThresholdP2N = 50;  ///< P到N档计数器阈值

constexpr double kSpeedCtrlInputThresholdN2D =
    0.06;  ///< N到D档速度控制输入阈值
constexpr double kSpeedCtrlInputThresholdN2P =
    0.05;  ///< N到P档速度控制输入阈值
constexpr double kSpeedCtrlInputThresholdD2N =
    0.05;  ///< D到N档速度控制输入阈值

constexpr double kAccelerationDeadZone = -0.05;  ///< 减速度死区

constexpr double kVehicleSpeedThresholdN2P = 0.01;  ///< N到P档车速阈值(m/s)
constexpr double kVehicleSpeedThresholdD2N = 0.1;   ///< D到N档车速阈值(m/s)
constexpr double kVehicleSpeedThresholdSpecial =
    0.25;  ///< 特殊条件车速阈值(m/s)

// 档位位置常量（来自CAN信号定义）
constexpr int kGearPositionNeutral = 0;    ///< N档位置
constexpr int kGearPositionReverse = 1;    ///< R档位置
constexpr int kGearPositionDriveMin = 2;   ///< D档最小位置
constexpr int kGearPositionDriveMax = 5;   ///< D档最大位置

constexpr uint32_t kEpbParkingBrakeStatus = 1U;  ///< EPB驻车制动激活状态值

constexpr double kMinAccForStarting = 0.06;  ///< 起步加速度阈值

// 坡度估计
constexpr double kSlopeLimitRad = 3 * kDeg2Rad;      ///< 坡度估计阈值
constexpr double kSlopeDeadZone = 1 * kDeg2Rad;      ///< 坡度死区
constexpr double kSlopeHoldThreshold = 2.5;  ///< 触发坡度保持策略的横向加速度阈值
constexpr unsigned int kSize = 25;                   ///< 坡度估计器历史大小
constexpr double kSpeedFilterCoef = 0.3;             ///< 车速滤波系数
constexpr double kImuAccFilterCoef = 0.2;            ///< IMU加速度滤波系数
constexpr double kSlopeFilterCoefOrdinary = 0.07;    ///< 普通工况坡度滤波系数
constexpr double kSlopeFilterCoefSpecial = 0.0105;   ///< 特殊工况坡度滤波系数
constexpr double kSlopeRateLimit = 5 * kDeg2Rad;     ///< 坡度变化率限制
constexpr double slope_filter_coef[kSize] = {
    0.003076923076923, 0.006153846153846, 0.009230769230769, 0.012307692307692,
    0.015384615384615, 0.018461538461538, 0.021538461538462, 0.024615384615385,
    0.027692307692308, 0.030769230769231, 0.033846153846154, 0.036923076923077,
    0.040000000000000, 0.043076923076923, 0.046153846153846, 0.049230769230769,
    0.052307692307692, 0.055384615384615, 0.058461538461538, 0.061538461538462,
    0.064615384615385, 0.067692307692308, 0.070769230769231, 0.073846153846154,
    0.076923076923077};

}  // namespace

// ?? 以下 using 声明中的顶层 namespace 拼写在截图 3 中形似 "rina"，但可能是其他类似字符（待核对实际源码） [3.png]
using rina::common::ErrorCode;
using rina::common::Status;
using rina::common::TrajectoryPoint;
using rina::common::VehicleStateProvider;
using rina::control::TrajectoryAnalyzer;
using rina::gaea::Time;

LonController::LonController()
    : counter_n2d_(kCounterThresholdN2D),  // 初始化计数器
      counter_d2n_(kCounterThresholdD2N),
      counter_n2p_(kCounterThresholdN2P),
      counter_p2n_(kCounterThresholdP2N) {
  ptr_accel_bias_est_ = std::make_shared<BiasEstimator>(accel_timewindow_);
  ptr_yawrate_bias_est_ = std::make_shared<BiasEstimator>(yawrate_timewindow_);
}

LonController::~LonController() = default;

std::string LonController::Name() const { return "LonController"; }

// 初始化控制器
/**
 * @brief 初始化纵向控制器
 * @param injector 依赖注入器
 * @param control_conf 控制参数配置
 * @return 初始化是否成功
 */
bool LonController::Init(std::shared_ptr<DependencyInjector> injector,
                         const rina::control::ControlParaConf *control_conf) {
  control_conf_ = control_conf;

  // 检查控制配置是否有效
  if (control_conf_ == nullptr) {
    controller_initialized_ = false;
    return false;
  }

  injector_ = injector;
  // 从控制配置中获取纵向控制器配置
  const LonControllerConf &lon_controller_conf =
      control_conf_->lon_controller_conf();

  double ts = lon_controller_conf.ts();  // 获取采样时间
  max_abs_speed_when_stopped_ =
      lon_controller_conf.max_abs_speed_when_stopped();
  a_preview_point_filt_coff_ = lon_controller_conf.a_preview_point_filt_coff();
  acc_cmd_use_preview_point_a_ =
      lon_controller_conf.acc_cmd_use_preview_point_a();
  preview_window_for_speed_pid_ =
      lon_controller_conf.preview_window_for_speed_pid();
  // 初始化速度PID控制器
  station_pid_controller_.Init(lon_controller_conf.station_pid_conf());
  speed_pid_controller_.Init(lon_controller_conf.low_speed_pid_conf());

  // 设置俯仰角数字滤波器
  SetDigitalFilterPitchAngle(lon_controller_conf);
  // 加载控制标定表
  LoadControlCalibrationTable();
  ctrl_enable_ = false;  // 初始化控制使能标志
  // 初始化计数器，设置阈值(kCounterThresholdN2D/kCounterThresholdD2N次对应1秒持续时间，控制周期0.02s)
  counter_n2d_ = Counter(kCounterThresholdN2D);
  counter_d2n_ = Counter(kCounterThresholdD2N);
  counter_n2p_ = Counter(kCounterThresholdN2P);
  counter_p2n_ = Counter(kCounterThresholdP2N);
  controller_initialized_ = true;  // 标记控制器已初始化
  acc_standstill_down_rate_ =
      control_conf_->acc_standstill_down_rate();  // 获取静止状态加速度下降率

  // 初始化低通滤波器
  std::array<double, 3> LowPassfilter_1_num = {0.0, 0.0, 1};
  std::array<double, 3> LowPassfilter_1_den = {0.0, 0.0, 1};
  LowPassfilter_1.Init(LowPassfilter_1_num, LowPassfilter_1_den);
  std::array<double, 3> LowPassfilter_Torque_num = {0.0, 0.15, 1};
  std::array<double, 3> LowPassfilter_Torque_den = {0.0, 0.15, 1};
  LowPassfilter_Torque.Init(LowPassfilter_Torque_num, LowPassfilter_Torque_den);

  return true;
};
void LonController::SetDigitalFilterPitchAngle(
    const LonControllerConf &lon_controller_conf) {
  double cutoff_freq =
      lon_controller_conf.pitch_angle_filter_conf().cutoff_freq();
  double ts = lon_controller_conf.ts();
  SetDigitalFilter(ts, cutoff_freq, &digital_filter_pitch_angle_);
}

/**
 * @brief 加载控制校准表
 *
 * 该方法加载以下校准表：
 *  - 加速度上升率表
 *  - 加速度下降率表
 *  - 加速度增益表
 *  - 加速度上限表
 *  - 加速度下限表
 */
void LonController::LoadControlCalibrationTable() {
  // 初始化各种1D查找表(用于加速度控制参数查询)
  acc_up_rate_table_ =
      std::make_unique<Lookup1D>();  // 加速度上升表(控制加速响应)
  acc_down_rate_table_ =
      std::make_unique<Lookup1D>();  // 加速度下降表(控制减速响应)
  acc_rate_gain_table_ =
      std::make_unique<Lookup1D>();  // 加速度增益表(根据车速调整增益)
  acc_up_lim_table_ =
      std::make_unique<Lookup1D>();  // 加速度上限表(限制最大加速度)
  acc_low_lim_table_ =
      std::make_unique<Lookup1D>();  // 加速度下限表(限制最小加速度)

  // 加载加速度上升率表数据
  for (auto info_acc_up_rate :
       control_conf_->acc_up_rate_table().acc_up_rate_info_()) {
    data2_2.push_back(
        std::make_pair(info_acc_up_rate.a_z1(), info_acc_up_rate.value()));
  }

  // 加载加速度下降率表数据
  for (auto info_acc_down_rate :
       control_conf_->acc_down_rate_table().acc_down_rate_info_()) {
    data2_3.push_back(
        std::make_pair(info_acc_down_rate.a_z1(), info_acc_down_rate.value()));
  }

  // 加载加速度增益表数据
  for (auto info_acc_rate_gain :
       control_conf_->acc_rate_gain_table().acc_rate_gain_info_()) {
    data2_4.push_back(std::make_pair(info_acc_rate_gain.vehspd(),
                                     info_acc_rate_gain.value()));
  }

  // 加载加速度上限表数据
  for (auto info_acc_up_lim :
       control_conf_->acc_up_lim_table().acc_up_lim_info_()) {
    data2_5.push_back(
        std::make_pair(info_acc_up_lim.vehspd(), info_acc_up_lim.value()));
  }

  // 加载加速度下限表数据
  for (auto info_acc_low_lim :
       control_conf_->acc_low_lim_table().acc_low_lim_info_()) {
    data2_6.push_back(
        std::make_pair(info_acc_low_lim.vehspd(), info_acc_low_lim.value()));
  }

  // 设置基础加速度上升率和下降率
  acc_up_rate_ = control_conf_->acc_up_rate();    // 默认加速度上升率
  acc_dwn_rate_ = control_conf_->acc_dwn_rate();  // 默认加速度下降率

  // 获取纵向控制器配置
}

void LonController::LoadControlCalibrationTable() {
  const LonControllerConf &lon_controller_conf =
      control_conf_->lon_controller_conf();
  // 获取控制标定表配置
  const auto &control_table = lon_controller_conf.calibration_table();
  // 准备插值数据
  Interpolation2D::DataType xyz;
  // 加载控制标定表数据
  for (const auto &calibration : control_table.calibration()) {
    xyz.push_back(std::make_tuple(calibration.speed(),
                                  calibration.acceleration(),
                                  calibration.command()));
  }
  // 初始化2D插值器(用于根据车速和加速度查询控制命令)
  control_interpolation_.reset(p: new Interpolation2D);
  ACHECK(control_interpolation_->Init(xyz))
      << "Fail to load control calibration table";
}
/**
 * @brief 重置控制器状态
 * @return 总是返回true
 */
bool LonController::Reset() {
  speed_pid_controller_.Reset();
  station_pid_controller_.Reset();
  return true;
}

/**
 * @brief 计算纵向控制命令
 * @param localization 定位信息
 * @param chassis_can 车辆CAN信号
 * @param planning_published_trajectory 规划轨迹
 * @param function_enable 功能使能标志
 * @param cmd 输出的控制命令
 * @return 控制命令计算是否成功
 */
bool LonController::ComputeControlCommand(
    const rina::localization::GlobalPose *localization,
    const udp::VehicleInfoToADU *chassis_can,
    const rina::planning::ADCTrajectory *planning_published_trajectory,
    const bool function_enable, rina::control::ControlCommand *cmd) {
  double torque_cmd_out = 0.0;
  // 更新输入数据
  localization_ = localization;
  chassis_can_ = chassis_can;
  ctrl_enable_ = function_enable;
  trajectory_message_ = planning_published_trajectory;

  if (trajectory_analyzer_ == nullptr
      || trajectory_analyzer_->seq_num()
          != trajectory_message_->header().sequence_num()) {
    trajectory_analyzer_.reset(p: new TrajectoryAnalyzer(planning_published_trajectory: trajectory_message_));
  }
  const rina::control::LonControllerConf &lon_controller_conf =
      control_conf_->lon_controller_conf();

  // 初始化调试信息
  auto debug = cmd->mutable_debug()->mutable_simple_lon_debug();
  debug->Clear();
  double vehicle_speed_kph =
      chassis_can_->vehicleinfoaddata().ad18fef100_ccvs_vehspd();

  double brake_cmd = 0.0;
  double throttle_cmd = 0.0;
  double acceleration_cmd_out = 0.0;

  // 获取时间参数
  double ts = lon_controller_conf.ts();
  double preview_time = lon_controller_conf.preview_window() * ts;
  double preview_time_for_speed_pid = preview_window_for_speed_pid_ * ts;

  // 计算误差 - 记录输入参数
  AINFO << "ComputeLongitudinalErrors inputs:"
        << "Vehicle position: x=" << localization_->position_enu().x()
        << ", y=" << localization_->position_enu().y()
        << " Heading: " << localization_->euler_angles().z()
        << " Preview time: " << preview_time << " Time step: " << ts;

  ComputeLongitudinalErrors(trajectory_analyzer: trajectory_analyzer_.get(), preview_time, ts,
                            debug);

  // 记录输出结果
  AINFO << "ComputeLongitudinalErrors outputs:"
        << "Station error: " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << debug->station_error()
        << ",Speed error: " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << debug->speed_error();
  AINFO << "preview_Station error: " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << debug->preview_station_error()
        << ",preview_Speed error: " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << debug->preview_speed_error()
        << ",preview_speed_reference: " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << debug->preview_speed_reference()
        << ",Acceleration error: " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << debug->acceleration_error();

  AINFO << "Matched point: x="
        << debug->current_matched_point().path_point().x()
        << ", y=" << debug->current_matched_point().path_point().y();
  double station_error_limit = lon_controller_conf.station_error_limit();
  double station_error_limited = 0.0;
  if (FLAGS_enable_speed_station_preview) {
    station_error_limited =
        rina::common::math::Clamp(value: debug->preview_station_error(),
                                  bound1: -station_error_limit, bound2: station_error_limit);
  } else {
    station_error_limited = rina::common::math::Clamp(
        value: debug->station_error(), bound1: -station_error_limit, bound2: station_error_limit);
  }

  if ((vehicle_speed_kph) > 1.0) {
    station_error_fnl = station_error_limited;
  } else if (((vehicle_speed_kph) <= 1.0) && (station_error_limited <= 0.25)) {
    station_error_fnl = std::fmin(x: 0.0, y: station_error_limited);
  } else if (((vehicle_speed_kph) <= 1.0) && (station_error_limited >= 0.8)) {
    station_error_fnl = station_error_limited;
  } else if (((vehicle_speed_kph) <= 1.0) && (station_error_fnl_pre <= 0.01)) {
    station_error_fnl = station_error_fnl_pre;
  } else {
    station_error_fnl = station_error_limited;
  }

  station_error_fnl_pre = station_error_fnl;

  // 计算速度控制器输入
  double speed_controller_input = 0.0;
  double speed_controller_input_limit =
      lon_controller_conf.speed_controller_input_limit();

  if (vehicle_speed_kph / kMps2Kph <= lon_controller_conf.switch_speed()) {
    speed_pid_controller_.SetPID(pid_conf: lon_controller_conf.low_speed_pid_conf());
  } else {
    speed_pid_controller_.SetPID(pid_conf: lon_controller_conf.high_speed_pid_conf());
  }

  double speed_offset = station_pid_controller_.Control(error: station_error_fnl, dt: ts);

  // 限制速度控制器输入范围
  double speed_controller_input_limited = 0.0;
  if (FLAGS_enable_speed_station_preview) {
    speed_controller_input = speed_offset + debug->preview_speed_error();
  } else {
    speed_controller_input = speed_offset + debug->speed_error();
  }
  speed_controller_input_limited = rina::common::math::Clamp(
      value: speed_controller_input, bound1: -speed_controller_input_limit,
      bound2: speed_controller_input_limit);

  // 速度PID控制: 计算闭环加速度指令
  double acceleration_cmd_closeloop = 0.0;
  acceleration_cmd_closeloop =
      speed_pid_controller_.Control(error: speed_controller_input_limited, dt: ts);
  debug->set_pid_saturation_status(
      value: speed_pid_controller_.IntegratorSaturationStatus());

  /**
   * @brief 坡度百分比值, 来自VCU12VCA_12_1BridgeLeftAngle信号
   * @unit 百分比 (%)
   * @range 坡度角度约[-20°, 20°]
   */
  double grade_percent =
      chassis_can_->vehicleinfoaddata().ad18ff0d27_vca_12_1bridgeleftangle();
  double slope_theta_rad_vcu = std::atan(x: grade_percent / 100.0);
  debug->set_acceleration_lookup(value: slope_theta_rad_vcu);
  AINFO << "slope_theta_rad_vcu = " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << slope_theta_rad_vcu;
  slope_theta_rad_vcu =
      std::clamp(val: slope_theta_rad_vcu, lo: -20.0 * kDeg2Rad, hi: 20.0 * kDeg2Rad);
  // 坡度补偿计算
  double slope_offset_compensation_vcu = digital_filter_pitch_angle_.Filter(
      x_insert: kGravitationalAcc * std::sin(x: slope_theta_rad_vcu));

  double slope_offset_compensation = 0.0;
  if (std::isfinite(x: slope_offset_compensation_vcu)) {
    slope_offset_compensation = slope_offset_compensation_vcu;
  } else {
    slope_offset_compensation = 0;
  }

  debug->set_slope_offset_compensation(value: slope_offset_compensation);

  // 计算最终加速度指令
  double acceleration_cmd =
      acceleration_cmd_closeloop
      + (acc_cmd_use_preview_point_a_
         * (debug->preview_acceleration_reference()))
      + (FLAGS_enable_slope_offset * debug->slope_offset_compensation());
  debug->set_is_full_stop(value: false);
  GetPathRemain(debug);

  // 添加调试信息
  AINFO << "Speed PID input (limited): " << speed_controller_input_limited;
  AINFO << "Speed PID output (closed-loop acc cmd): "
        << acceleration_cmd_closeloop;
  AINFO << "Slope compensation value: " << slope_offset_compensation;
  AINFO << "slope_theta_rad_vcu = " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << slope_theta_rad_vcu;
  AINFO << "Preview point acc reference: "
        << debug->preview_acceleration_reference();
  AINFO << "Slope compensation enabled: " << FLAGS_enable_slope_offset;
  AINFO << "acceleration_cmd: " << acceleration_cmd;

  double path_curv_far = cmd->path_curvature_far();
  AINFO << "path_curv_far=" << path_curv_far;
  acceleration_cmd_out = CalFinalAccCmd(&acceleration_cmd_raw: acceleration_cmd, &path_curvature_far: path_curv_far);
  static bool ctrl_enable_pre = false;
  bool ctrl_first_active = false;

  AINFO << "Control enable: " << ctrl_enable_;
  if (ctrl_enable_ && !ctrl_enable_pre) {
    ctrl_first_active = true;
    AINFO << "FIRST ACTIVE";
  }
  ctrl_enable_pre = ctrl_enable_;
  if (ctrl_first_active) {
    LowPassfilter_1.Reset();
    LowPassfilter_Torque.Reset();
    speed_pid_controller_.Reset();
    station_pid_controller_.Reset();
  }

  int current_gear_fb =
      chassis_can_->vehicleinfoaddata().ad18f101d0_tcu_currentgear();

  auto gear_req: GearPosition = rina::canbus::Chassis::GEAR_NEUTRAL;  // 默认空档
    GearControl(&: gear_req, current_gear_fb, veh_speed_mps, vehicle_speed_kph / kMps2Kph,
                acc_cmd_: acceleration_cmd, ctrl_enable_);
    // 统一设置档位
    cmd_->set_gear_location(value: gear_req);
    double torque_cmd_raw = 0;
    double estimated_road_slope = EstimateSlope(debug);  //< 估计的道路坡度
    if (current_gear_fb >= 2 && current_gear_fb <= 5) {  // D档
      torque_cmd_raw =
          CalFinalTorque(acceleration_cmd_: acceleration_cmd_out, estimated_road_slope, debug);
      torque_cmd_raw = LowPassfilter_Torque.Update(u: torque_cmd_raw);
    } else {
      torque_cmd_raw = 0;
    }

    if (!ctrl_enable_) {
      torque_cmd_out = 0;
    }

    if (acceleration_cmd_out > kAccelerationDeadZone) {
      torque_cmd_out = torque_cmd_raw;
      brake_cmd = 0;
      LowPassfilter_1.Reset();
    } else {
      torque_cmd_out = 0;
      LowPassfilter_Torque.Reset();
      acceleration_cmd_out = LowPassfilter_1.Update(u: acceleration_cmd_out);
      brake_cmd = acceleration_cmd_out;
    }

    AINFO << "Torque/Brake Command Debug: "
          << "vehicle_speed_kph: " << std::setw(n: 6) << std::fixed
          << std::setprecision(n: 3) << vehicle_speed_kph
          << ", acceleration_cmd_out: " << std::setw(n: 6) << std::fixed
          << std::setprecision(n: 3) << acceleration_cmd_out
          << ", calculated torque: " << std::setw(n: 6) << std::fixed
          << std::setprecision(n: 1) << torque_cmd_out;
    AINFO << "station_error_fnl = " << station_error_fnl;
    AINFO << "speed_offset = " << speed_offset;

    debug->set_station_error_limited(value: station_error_limited);
    debug->set_speed_offset(value: speed_offset);
    debug->set_speed_controller_input_limited(value: speed_controller_input_limited);
    debug->set_acceleration_cmd(value: acceleration_cmd);
    debug->set_throttle_cmd(value: throttle_cmd);
    debug->set_brake_cmd(value: brake_cmd);
    debug->set_speed_lookup(value: vehicle_speed_kph / kMps2Kph);
    debug->set_acceleration_cmd_closeloop(value: acceleration_cmd_closeloop);

    // 如果车辆加加速度驱动，则忽略油门和刹动命令
    cmd_->set_brake(value: brake_cmd);
    cmd_->set_target_torque(value: torque_cmd_out);

    AINFO << "ComputeControlCommand outputs:"
          << " - Torque cmd: " << std::setw(n: 6) << std::fixed
          << std::setprecision(n: 1) << torque_cmd_out
          << " - Brake cmd: " << std::setw(n: 6) << std::fixed
          << std::setprecision(n: 3) << brake_cmd
          << " - Throttle cmd: " << std::setw(n: 6) << std::fixed
          << std::setprecision(n: 3) << throttle_cmd
          << " - Acceleration cmd: " << std::setw(n: 6) << std::fixed
          << std::setprecision(n: 3) << acceleration_cmd_out
          << " - Gear location: " << cmd_->gear_location();

  return true;
}

/**
 * @brief 计算最终扭矩命令
 * @param acceleration_cmd 加速度指令
 * @return 计算得到的扭矩值
 *
 * 该方法考虑以下因素计算扭矩：
 * 1. 道路坡度估计
 * 2. 空气阻力计算
 * 3. 滚动阻力计算
 * 4. 坡度阻力计算
 * 5. 惯性力计算
 * 6. PI控制器补偿
 */
/**
 * @brief 计算最终加速度命令
 * @param accleration_cmd_raw 原始加速度指令
 * @param path_curvature_far 路径曲率
 * @return 处理后的加速度指令
 *
 * 处理逻辑：
 * 1. 加速度限制：
 *    - 根据车速查询加速度上下限
 *    - 考虑路径曲率调整限制值
 * 2. 变化率限制：
 *    - 应用加速度上升率限制
 *    - 应用加速度下降率限制
 *    - 低速特殊处理
 * 3. 输出处理：
 *    - 确保加速度平滑变化
 *    - 记录历史值用于下次计算
 *
 * 注意事项：
 *  - 低速时采用更保守的限制策略
 *  - 曲率较大时降低加速度限制
 */
double LonController::CalFinalAccCmd(double &acceleration_cmd_raw,
                                     double &path_curvature_far) {
  static double final_acc_out_pre = 0;  //< 上次加速度输出
  double final_acc_out;                 //< 最终加速度输出
  double final_acc_out_raw;             //< 原始加速度输出
  double final_acc_lowspd_sat;          //< 低速饱和加速度
  bool go_req = 0;                      //< 前进请求标志
  double acc_up_rate_lkup_raw;          //< 加速度上升率查找值
  double acc_dwn_rate_lkup_raw;         //< 加速度下降率查找值
  double acc_rate_gain;                 //< 加速度增益
  double acc_up_lim_lkup;               //< 加速度上限查找值
  double acc_up_rate_lkup;              //< 加速度上升率
  double acc_low_lim_lkup;              //< 加速度下限查找值
  double acc_dwn_rate_lkup;             //< 加速度下降率

  final_acc_out = 0;

  if (ctrl_enable_) {
    final_acc_out = acceleration_cmd_raw;
    control_en_timer_ = control_en_timer_ + 0.01;

    double veh_spd =
        std::fabs(x: chassis_can_->vehicleinfoaddata().ad18fef100_ccvs_vehspd());
    acc_up_rate_lkup_raw_ = acc_up_rate_table_->Interpolate(
        rows: data2_2.size(), table: data2_2, insig: final_acc_out_pre);
    acc_down_rate_lkup_raw_ = acc_down_rate_table_->Interpolate(
        rows: data2_3.size(), table: data2_3, insig: final_acc_out_pre);
    acc_rate_gain_ =
        acc_rate_gain_table_->Interpolate(rows: data2_4.size(), table: data2_4, insig: veh_spd);
    acc_up_lim_lkup_ =
        acc_up_lim_table_->Interpolate(rows: data2_5.size(), table: data2_5, insig: veh_spd);
    acc_up_rate_lkup_ = acc_up_rate_lkup_raw_ * acc_rate_gain_;
    acc_low_lim_lkup_ =
        acc_low_lim_table_->Interpolate(rows: data2_6.size(), table: data2_6, insig: veh_spd);

    AINFO << "Control table data: "
          << "acc_up_rate_lkup_raw_ size=" << data2_2.size()
          << ", acc_down_rate_lkup_raw_ size=" << data2_3.size()
          << ", acc_rate_gain_ size=" << data2_4.size()
          << ", acc_up_lim_lkup_ size=" << data2_5.size()
          << ", acc_low_lim_lkup_ size=" << data2_6.size()
          << ", vehspd=" << veh_spd
          << ", final_acc_out_pre=" << final_acc_out_pre;

    if (veh_spd < 1.5) {
      acc_dwn_rate_lkup_ = acc_standstill_down_rate_;
    } else {
      acc_dwn_rate_lkup_ = acc_dwn_rate_lkup_raw_;
    }
    if (path_curvature_far < -0.0075) {
      acc_up_lim_lkup_ = acc_up_lim_lkup_ * 0.75;
      acc_low_lim_lkup_ = acc_low_lim_lkup_ * 0.6;
    }

    if (acceleration_cmd_raw > acc_up_lim_lkup_) {
      final_acc_out_raw = acc_up_lim_lkup_;
    } else if (acceleration_cmd_raw < acc_low_lim_lkup_) {
      final_acc_out_raw = acc_low_lim_lkup_;
    } else {
      final_acc_out_raw = acceleration_cmd_raw;
    }

    if (((veh_spd) >= 0.2)) {
      final_acc_lowspd_sat = final_acc_out_raw;
    } else if ((final_acc_out_raw >= kMinAccForStarting) && ((veh_spd) < 0.2)) {
      final_acc_lowspd_sat = final_acc_out_raw;
    } else {
      final_acc_lowspd_sat = std::fmin(x: -0.05, y: final_acc_out_raw);
    }

    final_acc_out = rina::common::math::Clamp(
        value: final_acc_lowspd_sat, bound1: final_acc_out_pre + acc_dwn_rate_lkup_,
        bound2: final_acc_out_pre + acc_up_rate_lkup_);

    final_acc_out_pre = final_acc_out;
  } else {
    control_en_timer_ = 0;
    final_acc_out_pre = 0;
    final_acc_out = 0;
  }

  // 记录输出参数
  AINFO << "CalFinalAccCmd outputs: final_acc_out=" << std::setw(n: 6)
        << std::fixed << std::setprecision(n: 3) << final_acc_out
        << ", final_acc_out_pre=" << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << final_acc_out_pre
        << ", final_acc_out_raw=" << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << final_acc_out_raw
        << ", final_acc_lowspd_sat=" << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << final_acc_lowspd_sat
        << ", acc_low_lim_lkup_=" << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << acc_low_lim_lkup_
        << ", acc_up_lim_lkup_=" << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << acc_up_lim_lkup_
        << ", acc_dwn_rate_lkup_=" << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << acc_dwn_rate_lkup_
        << ", acc_up_rate_lkup_=" << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << acc_up_rate_lkup_;

  return final_acc_out;
}

double LonController::CalFinalTorque(
    double acceleration_cmd, double estimated_road_slope,
    rina::control::SimpleLongitudinalDebug *debug) {
  static double Last_Force_I = 0;                    //< 积分项历史值
  static double Last_sandstill_req = 0;              //< 静止请求历史值
  static bool Last_condition = false;                //< 条件状态历史值
  static bool Last_Torque_Request_Status = false;    //< 扭矩请求状态历史值
  double Road_Slope = 0;                             //< 道路坡度（弧度）
  bool standstill_req = 0;                           //< 静止请求标志
  constexpr double kair_dencity = 1.2041;            //< 空气密度（kg/m3）
  constexpr double kcoef_gravity = 9.81;             //< 重力系数（m/s2）
  double ego_speed =
      chassis_can_->vehicleinfoaddata().ad18fef100_ccvs_vehspd();  // km/h
  double ego_accel = chassis_can_->vehicleinfoaddata()
                         .ad18f0090b_vdc2_longitudinalacceleration();  // m/s2
  double coef_cd = FLAGS_coef_cd;
  double coef_rolling = FLAGS_coef_rolling;
  double coef_delta = FLAGS_coef_delta;
  double veh_wind_area = vehicle_param_.windward_area;
  double veh_mass = 9300;  // 默认卡车重量(kg)
  double vcu_weight =
      chassis_can_->vehicleinfoaddata().ad18fe7027_vcu_vehicleweight();
  if (!ctrl_enable_) {
    Last_Force_I = 0;
    Last_sandstill_req = 0;
  }
  // 重量合理范围检查(1-55吨)
  if (vcu_weight > 55000 || vcu_weight < 1000) {
  } else {
    veh_mass = vcu_weight;
  }
  double tyre_radius = vehicle_param_.wheel_rolling_radius;
  double trans_efficiency = vehicle_param_.transmission_efficiency;

  // 道路坡度估计参数
  static double ego_spd_history[10] = {[0]=0};      //< 车速历史记录(km/h)
  static double Last_road_slope = 0;                //< 上次道路坡度
  double percent = 0;                                //< 百分比系数
  double FLAGS_slope_max_duration = 4000;            //< 坡度最大持续时间(ms)

  static unsigned int Timer0 = 0;
  for (int i = 9; i >= 1; i--) {
    ego_spd_history[i] = ego_spd_history[i - 1];
  }
  ego_spd_history[0] = ego_speed / kMps2Kph;
  Timer0 = Timer0 + 1;
  if (Timer0 > 10) {
    Timer0 = 10;
  }
  // 根据速度历史计算加速度
  double temp = (ego_spd_history[0] - ego_spd_history[5])
              + (ego_spd_history[1] - ego_spd_history[6])
              + (ego_spd_history[2] - ego_spd_history[7])
              + (ego_spd_history[3] - ego_spd_history[8])
              + (ego_spd_history[4] - ego_spd_history[9]);
  // 临时变量值
  double ego_accel_dvdt = 0;
  ego_accel_dvdt =
      temp / (25 * control_conf_->control_period());  // 需要检查计算
  // 道路坡度估计
  if (Timer0 < 10) {
    Road_Slope = 0;
  } else {
    Road_Slope = (ego_accel - ego_accel_dvdt) / kcoef_gravity;
  }

  // 坡度估计完成
  // estimated_road_slope = Road_Slope;

  // 道路坡度估计结束
  if (FLAGS_enable_slope_estimate) {
    slope_ = std::clamp(
        val: estimated_road_slope, lo: -kSlopeLimitRad,
        hi: kSlopeLimitRad);  // todo:need to fix the slope estimation logic
  } else {
    slope_ = 0.;
  }
  AINFO << "estimated_road_slope: " << estimated_road_slope
        << ", FLAGS_enable_slope_estimate: " << FLAGS_enable_slope_estimate
        << ", slope_: " << slope_;

  // 计算各种力分量，使用已验证的车辆质量
  double Force_Air = 0.5 * coef_cd * kair_dencity * veh_wind_area
      * (ego_speed / kMps2Kph) * (ego_speed / kMps2Kph);

  // 滚动阻力计算，使用已验证的质量
  double Force_Rolling = coef_rolling * veh_mass * kcoef_gravity * cos(x: slope_);

  // 坡度阻力计算，使用已验证的质量
  double Force_Slope = veh_mass * kcoef_gravity * sin(x: slope_);

  // 惯性力计算，使用已验证的质量
  double Force_Inertia = coef_delta * veh_mass * acceleration_cmd;

  // 总阻力计算
  double Force_Resist = Force_Air + Force_Inertia + Force_Rolling + Force_Slope;

  // 添加调试日志
  AINFO << "Force calculations with mass=" << std::setw(n: 5) << std::fixed
        << std::setprecision(n: 0) << veh_mass << " kg,"
        << "Air: " << std::setw(n: 6) << std::fixed << std::setprecision(n: 1)
        << Force_Air << " N,"
        << "Rolling: " << std::setw(n: 6) << std::fixed << std::setprecision(n: 1)
        << Force_Rolling << " N,"
        << "Slope: " << std::setw(n: 6) << std::fixed << std::setprecision(n: 1)
        << Force_Slope << " N,"
        << "Inertia: " << std::setw(n: 6) << std::fixed << std::setprecision(n: 1)
        << Force_Inertia << " N,"
        << "Total Resist: " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 1) << Force_Resist << " N";

  double Error_Ax = acceleration_cmd - ego_accel;
  double Force_PI = 0;
  double K_P_Ax2Trq = FLAGS_accel_to_torque_kp;
  double K_I_Ax2Trq = FLAGS_accel_to_torque_ki;
  double Force_P = K_P_Ax2Trq * (Error_Ax);
  double Force_I = K_I_Ax2Trq * (Error_Ax);


  Last_Force_I = Last_Force_I + Force_I;
  AINFO << "PI Controller forces:"
        << " - P_force: " << std::setw(n: 6) << std::fixed << std::setprecision(n: 1)
        << Force_P << " - I_force: " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 1) << Last_Force_I;
  // 根据当前挡位选择传动比
  double current_gear =
      chassis_can_->vehicleinfoaddata().ad18f101d0_tcu_currentgear();
  double trans_ratio = 1;  // 默认使用基础传动比

  // 使用车辆参数中的具体挡位传动比
  if (current_gear == 1) {  // R档
    trans_ratio = vehicle_param_.transmission_ratio_R;
  } else if (current_gear == 2) {  // D1档
    trans_ratio = vehicle_param_.transmission_ratio_D1;
  } else if (current_gear == 3) {  // D2档
    trans_ratio = vehicle_param_.transmission_ratio_D2;
  } else if (current_gear == 4) {  // D3档
    trans_ratio = vehicle_param_.transmission_ratio_D3;
  } else if (current_gear == 5) {  // D4档
    trans_ratio = vehicle_param_.transmission_ratio_D4;
  }

  double Torque_Combustion =
      (Force_P + Force_Resist) * tyre_radius / trans_efficiency / trans_ratio;
  // 限制燃烧扭矩幅值
  if (Torque_Combustion >= FLAGS_torque_combustion_upper_limit) {
    Torque_Combustion = FLAGS_torque_combustion_upper_limit;
  }
  if (Torque_Combustion <= FLAGS_torque_combustion_lower_limit) {
    Torque_Combustion = FLAGS_torque_combustion_lower_limit;
  }

  AINFO << "current_gear: " << current_gear;
  AINFO << "Torque Before Limit: " << Torque_Combustion << " Nm";

  double Torque_Target_Out = 0;
  if (acceleration_cmd > kAccelerationDeadZone) {
    Torque_Target_Out = Torque_Combustion;
  }

  AINFO << "Final Torque Output: " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 1) << Torque_Target_Out << " Nm"
        << ", K_P_Ax2Trq = " << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 3) << K_P_Ax2Trq
        << ", Torque compute:" << std::setw(n: 6) << std::fixed
        << std::setprecision(n: 1) << Torque_Combustion;
  return Torque_Target_Out;
}

double LonController::EstimateSlope(
    rina::control::SimpleLongitudinalDebug *debug) {
  double raw_ego_speed_mps =
      chassis_can_->vehicleinfoaddata().ad18fef100_ccvs_vehspd()
      / kMps2Kph;  // mps
  double raw_ego_accel_imu =
      chassis_can_->vehicleinfoaddata()
      .ad18f0090b_vdc2_longitudinalacceleration();  // m/s2
  double ego_yawrate_rads_raw =
      chassis_can_->vehicleinfoaddata().ad18f0090b_vdc2_yawrate();  // rad/s
  double ego_steerwheel_angle =
      chassis_can_->vehicleinfoaddata().ad18ffe313_steering_angle();  // deg

  double road_slope = 0.0;  ///< 道路坡度(弧度)
  static unsigned int timer = 0;
  static double ego_accel_wheel_history[kSize] = {
      [0]=0};                                     //<实测加速度历史记录(mps2)
  static double ego_accel_imu_history[kSize] = {[0]=0};  //<imu加速度历史记录(mps2)
  static double ego_yawrate_rads_history[kSize] = {
      [0]=0};  //<yawrate历史记录(radps)

  for (int i = kSize - 1; i >= 1; i--) {
    ego_accel_wheel_history[i] = ego_accel_wheel_history[i - 1];
    ego_accel_imu_history[i] = ego_accel_imu_history[i - 1];
    ego_yawrate_rads_history[i] = ego_yawrate_rads_history[i - 1];
  }
  // 更新车辆实际加速度
  double delta_raw_ego_speed_mps =
      raw_ego_speed_mps - previous_raw_ego_speed_mps_;
  delta_raw_ego_speed_mps = std::clamp(val: delta_raw_ego_speed_mps, lo: -0.18, hi: 0.1);
  raw_ego_speed_mps = previous_raw_ego_speed_mps_ + delta_raw_ego_speed_mps;
  // if (delta_raw_ego_speed_mps > 0.1 || delta_raw_ego_speed_mps < -0.18) {
  //   raw_ego_speed_mps = previous_raw_ego_speed_mps_;
  // }
  previous_raw_ego_speed_mps_ = raw_ego_speed_mps;
  double ego_speed_mps = (kSpeedFilterCoef * raw_ego_speed_mps)
                       + ((1 - kSpeedFilterCoef) * previous_ego_speed_mps_);
  double current_ego_accel_wheel = (ego_speed_mps - previous_ego_speed_mps_)
                                 / control_conf_->control_period();
  ego_accel_wheel_history[0] = current_ego_accel_wheel;
  previous_ego_speed_mps_ = ego_speed_mps;

  // 更新IMU纵向加速度
  double current_ego_accel_imu =
      (kImuAccFilterCoef * raw_ego_accel_imu)
      + (1 - kImuAccFilterCoef) * previous_raw_ego_accel_imu_;
  ego_accel_imu_history[0] = current_ego_accel_imu;
  previous_raw_ego_accel_imu_ = current_ego_accel_imu;

  // 更新横摆角速度
  ego_yawrate_rads_history[0] = ego_yawrate_rads_raw;

  // 计数器
  timer = timer + 1;
  if (timer > kSize) {
    timer = kSize;
  }

  double ego_accel_wheel =
      std::inner_product(first1: slope_filter_coef, last1: slope_filter_coef + kSize,
                         first2: ego_accel_wheel_history, init: 0.0);
  double ego_accel_imu = std::inner_product(
      first1: slope_filter_coef, last1: slope_filter_coef + kSize, first2: ego_accel_imu_history, init: 0.0);
  double ego_yawrate_rads =
      std::inner_product(first1: slope_filter_coef, last1: slope_filter_coef + kSize,
                         first2: ego_yawrate_rads_history, init: 0.0);

  // 传感器零偏估计
  if (std::abs(x: ego_speed_mps) < 1e-2) {
    ptr_accel_bias_est_->AddData(data: ego_accel_imu);
    if (ptr_accel_bias_est_->GetDataCount() >= accel_timewindow_) {
      accel_imu_bias_ = ptr_accel_bias_est_->GetBias();
      AINFO << "[Control] accel_imu bias est success! accel_imu bias:"
            << accel_imu_bias_;
    }
  ptr_yawrate_bias_est_->AddData(data: ego_yawrate_rads);
  if (ptr_yawrate_bias_est_->GetDataCount() >= yawrate_timewindow_) {
    yawrate_bias_ = ptr_yawrate_bias_est_->GetBias();
    AINFO << "[Control] yawrate bias est success! yawrate bias:"
          << yawrate_bias_;
  }
}
ego_accel_imu -= accel_imu_bias_;
ego_yawrate_rads -= yawrate_bias_;

debug->set_speed_p(value: ego_accel_wheel);
debug->set_speed_i(value: ego_accel_imu);

// 特殊工况识别
static double ego_speed_mps_history[10] = {[0]=0};  // 车速历史记录(mps)
for (int i = 9; i >= 1; i--) {
  ego_speed_mps_history[i] = ego_speed_mps_history[i - 1];
}
ego_speed_mps_history[0] = ego_speed_mps;
double sum_ego_speed_mps_history =
    std::accumulate(first: ego_speed_mps_history, last: ego_speed_mps_history + 10, init: 0.0);

static double ego_accel_wheel_final_history[5] = {[0]=0};  // 加速度历史记录(mps2)
for (int i = 4; i >= 1; i--) {
  ego_accel_wheel_final_history[i] = ego_accel_wheel_final_history[i - 1];
}
ego_accel_wheel_final_history[0] = ego_accel_wheel;
double sum_ego_accel_wheel_final_history = std::accumulate(
    first: ego_accel_wheel_final_history, last: ego_accel_wheel_final_history + 5, init: 0.0);

bool is_rapid_acceleration =
    (sum_ego_speed_mps_history > 1.0)
    && (std::fabs(x: sum_ego_accel_wheel_final_history) > 1.5);
is_rapid_acceleration =
    rapid_acceleration_debouncer_.Update(raw_signal: is_rapid_acceleration);

bool is_brake_to_standstill = (sum_ego_speed_mps_history < 5.0)
                              && (sum_ego_accel_wheel_final_history < -1.5);
is_brake_to_standstill =
    brake_to_standstill_debouncer_.Update(raw_signal: is_brake_to_standstill);

bool is_standstill_to_move = (sum_ego_speed_mps_history < 1.0)
                             && (sum_ego_accel_wheel_final_history > 0.5);
is_standstill_to_move =
    standstill_to_move_debouncer_.Update(raw_signal: is_standstill_to_move);

bool is_standstill_ = (sum_ego_speed_mps_history < 0.5)
                     && (sum_ego_accel_wheel_final_history < 0.2)
                     && (!is_brake_to_standstill)
                     && (!is_standstill_to_move);

bool isSpecialScene = is_rapid_acceleration || is_brake_to_standstill
                      || is_standstill_to_move || is_standstill_;
// debug->set_current_station(isSpecialScene);
// 道路坡度估计
if (timer < kSize) {
  return 0.0;
} else {
  double est_ay = std::abs(x: ego_speed_mps * ego_yawrate_rads);  // 横向加速度
  // debug->set_current_speed(est_ay);
  if (est_ay > kSlopeHoldThreshold) {
    return previous_road_slope_;  // 保持策略，维持上一周期的坡度值
  }

  // 计算坡度，方法一
  // double front_wheel_angle =
  //     ego_steerwheel_angle / vehicle_param_.steer_ratio;
  // double beta = (vehicle_param_.rear_axle_to_cg * front_wheel_angle)
  //               / vehicle_param_.wheel_base;
  // double v_y = ego_speed_mps * beta;
  // double coriolis_acc = ego_yawrate_rads * v_y;  // 科氏加速度
  // double g_sin_theta = ego_accel_imu - ego_accel_wheel + coriolis_acc;
  // g_sin_theta = std::clamp(g_sin_theta, -2.0, 2.0);
  // double raw_slope = std::asin(g_sin_theta / kGravitationalAcc);

  // 计算坡度，方法二（港口）
  double g_sin_theta =
      ego_accel_imu
      + vehicle_param_.yawsensor_imu_L
            * std::cos(x: vehicle_param_.yawsensor_imu_theta) * ego_yawrate_rads
      - ego_accel_wheel
      - (ego_speed_mps
         + vehicle_param_.yawsensor_imu_L
               * std::sin(x: vehicle_param_.yawsensor_imu_theta)
               * ego_yawrate_rads)
            / std::max(a: 0.1, b: ego_speed_mps);
  g_sin_theta = std::clamp(val: g_sin_theta, lo: -2.0, hi: 2.0);
  double raw_slope = std::asin(x: g_sin_theta / kGravitationalAcc);

  double delta_raw_slope =
      std::clamp(val: raw_slope - previous_raw_slope_,
                 lo: -kSlopeRateLimit * control_conf_->control_period(),
                 hi: kSlopeRateLimit * control_conf_->control_period());
  raw_slope = previous_raw_slope_ + delta_raw_slope;
  previous_raw_slope_ = raw_slope;
  // 滤波
  if (isSpecialScene) {
    road_slope = (kSlopeFilterCoefSpecial * raw_slope)
                 + ((1 - kSlopeFilterCoefSpecial) * previous_road_slope_);
  } else {
    road_slope = (kSlopeFilterCoefOrdinary * raw_slope)
                 + ((1 - kSlopeFilterCoefOrdinary) * previous_road_slope_);
  }
  previous_road_slope_ = road_slope;

  if (std::fabs(x: road_slope) < kSlopeDeadZone) {
    road_slope = 0.0;
  }
  road_slope = std::clamp(val: road_slope, lo: -kSlopeLimitRad, hi: kSlopeLimitRad);

  // debug->set_current_jerk(raw_slope);
  debug->set_speed_d(value: road_slope);
  return road_slope;
}
}

void LonController::ComputeLongitudinalErrors(
    const TrajectoryAnalyzer *trajectory_analyzer, const double preview_time,
    const double ts, rina::control::SimpleLongitudinalDebug *debug) {
  // 车辆运动在Frenet坐标系下的分解
  // s:    沿参考轨迹的纵向累计距离
  // s_dot: 沿参考轨迹的纵向速度
  // d:    相对于参考轨迹的横向距离
  // d_dot: 横向距离变化率，即dd/dt
  double s_matched = 0.0;
  double s_dot_matched = 0.0;
  double d_matched = 0.0;
  double d_dot_matched = 0.0;
  double vehicle_speed_kph =
      chassis_can_->vehicleinfoaddata().ad18fef100_ccvs_vehspd();
  auto matched_point /*: common::PathPoint*/ = trajectory_analyzer->QueryMatchedPathPoint(
      x: localization_->position_enu().x(), y: localization_->position_enu().y());
  AINFO << "matched_point.s(): " << matched_point.s();
  double yaw_rad =
      localization_->euler_angles().z() * kDeg2Rad;  // 将度转换为弧度
  trajectory_analyzer->ToTrajectoryFrame(
      x: localization_->position_enu().x(), y: localization_->position_enu().y(),
      theta: yaw_rad, v: vehicle_speed_kph / kMps2Kph, matched_point, ptr_s: &s_matched,
      ptr_s_dot: &s_dot_matched, ptr_d: &d_matched, ptr_d_dot: &d_dot_matched);

  double current_control_time =
      trajectory_message_->header().measurement_time();
  int64_t milliseconds =
      static_cast<int64_t>(current_control_time * 1000) % 1000;
  double preview_control_time = current_control_time + preview_time;
  int64_t preview_milliseconds =
      static_cast<int64_t>(preview_control_time * 1000) % 1000;
  AINFO << "Preview control timestamp: "
        << Time(seconds: preview_control_time).ToString() << " (" << std::fixed
        << std::setprecision(n: 3) << preview_control_time << " seconds, "
        << preview_milliseconds << " ms)";

  double preview_control_speed_time =
      current_control_time + preview_time_for_speed_pid;

  TrajectoryPoint reference_point =
      trajectory_analyzer->QueryNearestPointByAbsoluteTime(
          t: current_control_time);
  TrajectoryPoint preview_point =
      trajectory_analyzer->QueryNearestPointByAbsoluteTime(
          t: preview_control_time);
  TrajectoryPoint preview_point_speed =
      trajectory_analyzer->QueryNearestPointByAbsoluteTime(
          t: preview_control_speed_time);

  debug->mutable_current_matched_point()->mutable_path_point()->set_x(
      value: matched_point.x());
  debug->mutable_current_matched_point()->mutable_path_point()->set_y(
      value: matched_point.y());
  debug->mutable_current_reference_point()->mutable_path_point()->set_x(
      value: reference_point.path_point().x());
  debug->mutable_current_reference_point()->mutable_path_point()->set_y(
      value: reference_point.path_point().y());
  debug->mutable_preview_reference_point()->mutable_path_point()->set_x(
      value: preview_point.path_point().x());
  debug->mutable_preview_reference_point()->mutable_path_point()->set_y(
      value: preview_point.path_point().y());

  double heading_error =
      rina::common::math::NormalizeAngle(angle: yaw_rad - matched_point.theta());
  double lon_speed = vehicle_speed_kph / kMps2Kph * std::cos(x: heading_error);
  double lon_acceleration = chassis_can_->vehicleinfoaddata()
                                .ad18f0090b_vdc2_longitudinalacceleration()
                            * std::cos(x: heading_error);
  double one_minus_kappa_lat_error = 1
                                     - reference_point.path_point().kappa()
                                           * (vehicle_speed_kph / kMps2Kph)
                                           * std::sin(x: heading_error);

  debug->set_station_reference(value: reference_point.path_point().s());
  debug->set_current_station(value: s_matched);
  debug->set_station_error(value: reference_point.path_point().s() - s_matched);
  debug->set_speed_reference(value: reference_point.v());
  debug->set_current_speed(value: lon_speed);
  debug->set_speed_error(value: reference_point.v() - s_dot_matched);
  debug->set_acceleration_reference(value: reference_point.a());
  debug->set_current_acceleration(value: lon_acceleration);
  debug->set_acceleration_error(value: reference_point.a()
                                - lon_acceleration / one_minus_kappa_lat_error);

  AINFO << "Reference Point:"
        << "s=" << std::setw(n: 8) << std::fixed << std::setprecision(n: 2)
        << reference_point.path_point().s() << " m, "
        << "v=" << std::setw(n: 5) << std::fixed << std::setprecision(n: 2)
        << reference_point.v() << " m/s, "
        << "a=" << std::setw(n: 5) << std::fixed << std::setprecision(n: 2)
        << reference_point.a() << " m/s² ";
  AINFO << "preview_point:"
        << "s=" << std::setw(n: 8) << std::fixed << std::setprecision(n: 2)
        << preview_point.path_point().s() << " m, "
        << "v=" << std::setw(n: 5) << std::fixed << std::setprecision(n: 2)
        << preview_point.v() << " m/s, "
        << "a=" << std::setw(n: 5) << std::fixed << std::setprecision(n: 2)
        << preview_point.a() << " m/s², "
        << "relative_time= " << std::setw(n: 5) << std::fixed
        << std::setprecision(n: 2) << preview_point.relative_time();

  AINFO << "Current State:"
        << "s_matched=" << std::setw(n: 8) << std::fixed << std::setprecision(n: 2)
        << s_matched << " m, "
        << "lon_speed=" << std::setw(n: 5) << std::fixed << std::setprecision(n: 2)
        << lon_speed << " m/s, "
        << "lon_acceleration=" << std::setw(n: 5) << std::fixed
        << std::setprecision(n: 2) << lon_acceleration << " m/s², "
        << "s_dot_matched=" << std::setw(n: 5) << std::fixed
        << std::setprecision(n: 2) << s_dot_matched << " m/s";

  double jerk_reference =
      (debug->acceleration_reference() - previous_acceleration_reference_) / ts;
  double lon_jerk =
      (debug->current_acceleration() - previous_acceleration_) / ts;
  debug->set_jerk_reference(value: jerk_reference);
  debug->set_current_jerk(value: lon_jerk);
  debug->set_jerk_error(value: jerk_reference - lon_jerk, one_minus_kappa_lat_error);  // ?? 第二个参数 one_minus_kappa_lat_error 语义可疑，但截图显示如此 [36.png L1137]
  previous_acceleration_reference_ = debug->acceleration_reference();
  previous_acceleration_ = debug->current_acceleration();

  if ((vehicle_speed_kph) > 0.5) {
    preview_point_v_cut_ = preview_point_speed.v();
  } else if (((vehicle_speed_kph) <= 0.5) && (preview_point_speed.v() <= 0.3)) {
    preview_point_v_cut_ = 0.0;
  } else {
    preview_point_v_cut_ = preview_point_speed.v();
  }

  preview_point_v_filt_ =
      (preview_point_v_cut_ * 0.1) + ((1 - 0.1) * preview_point_v_pre_);

  preview_point_v_pre_ = preview_point_v_filt_;

  debug->set_preview_station_error(value: preview_point.path_point().s() - s_matched);
  debug->set_preview_speed_error(value: preview_point_v_filt_ - s_dot_matched);
  debug->set_preview_speed_reference(value: preview_point_v_filt_);

  preview_point_a_filt_ =
      (preview_point_speed.a() * a_preview_point_filt_coff_)
      + ((1 - a_preview_point_filt_coff_) * preview_point_a_pre_);

  preview_point_a_pre_ = preview_point_a_filt_;

  debug->set_preview_acceleration_reference(value: preview_point_a_filt_);
}

void LonController::SetDigitalFilter(
    double ts, double cutoff_freq,
    rina::common::DigitalFilter *digital_filter) {
  std::vector<double> denominators;
  std::vector<double> numerators;
  rina::common::LpfCoefficients(ts, cutoff_freq, denominators: &denominators, numerators: &numerators);
  digital_filter->set_coefficients(denominators, numerators);
}

void LonController::GetPathRemain(
    rina::control::SimpleLongitudinalDebug *debug) {
  int stop_index = 0;
  static constexpr double kSpeedThreshold = 1e-3;
  static constexpr double kForwardAccThreshold = -1e-2;
  static constexpr double kBackwardAccThreshold = 1e-1;
  static constexpr double kParkingSpeed = 0.1;

  if (trajectory_message_->gear() == rina::canbus::Chassis::GEAR_DRIVE) {
    while (stop_index < trajectory_message_->trajectory_point_size()) {
      auto &current_trajectory_point =
          trajectory_message_->trajectory_point(index: stop_index);
      if (fabs(x: current_trajectory_point.v()) < kSpeedThreshold
          && current_trajectory_point.a() > kForwardAccThreshold
          && current_trajectory_point.a() < 0.0) {
        break;
      }
      ++stop_index;
    }
  } else {
    while (stop_index < trajectory_message_->trajectory_point_size()) {
      auto &current_trajectory_point =
          trajectory_message_->trajectory_point(index: stop_index);
      if (current_trajectory_point.v() < kSpeedThreshold
          && current_trajectory_point.a() < kBackwardAccThreshold
          && current_trajectory_point.a() > 0.0) {
        break;
      }
      ++stop_index;
    }
  }
  if (stop_index == trajectory_message_->trajectory_point_size()) {
    --stop_index;
    if (fabs(x: trajectory_message_->trajectory_point(index: stop_index).v())
        < kParkingSpeed) {
      AINFO << "Selected last point as stop point";
    } else {
      AINFO << "Found last point in path with speed above deadzone";
    }
  }
  debug->set_path_remain(
      value: trajectory_message_->trajectory_point(index: stop_index).path_point().s()
      - debug->current_station());
}

void LonController::GearControl(rina::canbus::Chassis::GearPosition &gear_req,
                                 const int current_gear_fb,
                                 const double veh_speed_mps,
                                 const double acc_cmd, const bool ctrl_enable_) {
  AINFO << "GearControl: current_gear_fb=" << current_gear_fb
        << ", veh_speed_mps=" << veh_speed_mps << ", acc_cmd=" << acc_cmd
        << ", ctrl_enable=" << ctrl_enable_;

  if (current_gear_fb == kGearPositionNeutral) {
    gear_req = rina::canbus::Chassis::GEAR_NEUTRAL;
  } else if (current_gear_fb == kGearPositionReverse) {
    gear_req = rina::canbus::Chassis::GEAR_REVERSE;
  } else if (current_gear_fb >= kGearPositionDriveMin
             && current_gear_fb <= kGearPositionDriveMax) {
    gear_req = rina::canbus::Chassis::GEAR_DRIVE;
  } else {
    AINFO << "Invalid Gear Feedback: " << current_gear_fb;
    gear_req = rina::canbus::Chassis::GEAR_NEUTRAL;
  }

  // TODO: P挡到D挡需要增加N挡的中间状态，这适应更多车型（当前车型没有P挡；目前也没有R挡。
  // 档位控制
  bool n2d_condition = acc_cmd > kSpeedCtrlInputThresholdN2D && ctrl_enable_;
  counter_n2d_.Update(condition: n2d_condition);  // 当速度控制输入>kSpeedCtrlInputThresholdN2D时增加计数，否则减少
  AINFO << "N2D counter: condition=" << n2d_condition
        << ", count=" << counter_n2d_.GetCount();
  bool n2p_condition =
      (current_gear_fb == kGearPositionNeutral && ctrl_enable_
       && acc_cmd < kSpeedCtrlInputThresholdN2P
       && std::fabs(x: veh_speed_mps) <= kVehicleSpeedThresholdN2P);
  counter_n2p_.Update(condition: n2p_condition);
  AINFO << "N2P counter: condition=" << n2p_condition
        << ", count=" << counter_n2p_.GetCount();
  // 特殊条件覆盖
  if (current_gear_fb == kGearPositionNeutral && ctrl_enable_
      && std::fabs(x: veh_speed_mps) <= kVehicleSpeedThresholdSpecial
      && counter_n2d_.IsTriggered()) {
    gear_req = rina::canbus::Chassis::GEAR_DRIVE;
    AINFO << "Shift D Gear \n";
    if (current_gear_fb >= kGearPositionDriveMin
        && current_gear_fb <= kGearPositionDriveMax) {
      counter_n2d_.Reset();  // D挡切换成功后重置D挡计数器
    }
    counter_d2n_.Reset();  // 同时重置N挡计数器
  }
  // N挡控制逻辑
  bool d2n_condition = (acc_cmd < kSpeedCtrlInputThresholdD2N
                        && std::fabs(x: veh_speed_mps) <= kVehicleSpeedThresholdD2N
                        && ctrl_enable_);
  counter_d2n_.Update(condition: d2n_condition);
  AINFO << "D2N counter: condition=" << d2n_condition
        << ", count=" << counter_d2n_.GetCount();
  if (counter_d2n_.IsTriggered()) {
    gear_req = rina::canbus::Chassis::GEAR_NEUTRAL;
    AINFO << "Counter triggered, shift to N gear";
    if (current_gear_fb == kGearPositionNeutral) {
      counter_d2n_.Reset();  // N挡切换成功后重置N挡计数器
    }
    counter_n2d_.Reset();  // 同时重置D挡计数器
  }
  if (!n2p_condition) {
    counter_n2p_.Reset();
  }
  if (counter_n2p_.IsTriggered()) {
    gear_req = rina::canbus::Chassis::GEAR_PARKING;
    AINFO << "Counter triggered, shift to P gear";
    if (static_cast<uint32_t>(
            chassis_can_->vehicleinfoadata().adcff9e50_epb_parkbrkst())
        == static_cast<uint32_t>(kEpbParkingBrakeStatus)) {
      counter_n2p_.Reset();
    }
  }
}

}  // namespace control
}  // namespace mlp
