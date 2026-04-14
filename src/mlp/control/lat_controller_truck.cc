#include "controller/lat_controller_truck.h"
#include "ads_common/math/math_utils.h"

using rina::control::SimpleLateralDebug;
using rina::control::TrajectoryAnalyzer;

namespace mlp {
namespace control {
namespace {
constexpr double kMps2Kph = 3.6;
constexpr double kMaxSpeed = 100.0;
constexpr double kLow = 1.0;
constexpr double kh = 0;  // ?? constant name / value unclear (possibly kLh / kTh) [1.png]
constexpr double kRate_limit_fb = 120.0;
constexpr double kRate_limit_ff = 165.0;
constexpr double kRate_limit_total = 300.0;
constexpr double kMin_prev_dist = 4.0;
constexpr double kMin_reach_dis_theta = 3.8;
constexpr double kPideg = 180;
}  // namespace

LatControllerTruck::LatControllerTruck() : name_("Truck Lateral Controller") {}

LatControllerTruck::~LatControllerTruck() = default;

bool LatControllerTruck::LoadControlConf(
    const std::shared_ptr<ControlParaConf> control_conf) {
  if (!control_conf) {
    AERROR << "[LatControllerTruck] control_conf == nullptr";
    return false;
  }

  for (auto info_max_theta_deg : control_conf_->lat_controller_conf()  // ?? lines 27-35 folded in [2.png]; if-branch + first for-loop header reconstructed
                                     .yawrate_gain_table()
                                     .yawrate_gain_info()) {
    data_max_theta_deg_.push_back(std::make_pair(info_max_theta_deg.speed(),
                                                 info_max_theta_deg.yawrate()));
  }
  for (auto info_prev_time_dist : control_conf_->lat_controller_conf()
                                      .theta_yawrate_gain_table()
                                      .theta_yawrate_gain_info()) {
    data_prev_time_dist_.push_back(std::make_pair(info_prev_time_dist.speed(),
                                                  info_prev_time_dist.value()));
  }
  for (auto info_reach_time_theta : control_conf_->lat_controller_conf()
                                        .theta_yawrate_gain_table2()
                                        .theta_yawrate_gain_info2()) {
    data_reach_time_theta_.push_back(std::make_pair(
        info_reach_time_theta.ang_req(), info_reach_time_theta.statio_wheel()));  // ?? 'statio_wheel' may be 'status_wheel' [2.png]
  }
  for (auto info_prew_time_dt_theta : control_conf_->lat_controller_conf()
                                          .end_pnt_time_table()
                                          .end_pnt_time_info()) {
    data_prew_time_dt_theta_.push_back(std::make_pair(
        info_prew_time_dt_theta.curvature(), info_prew_time_dt_theta.value()));
  }
  for (auto info_near_point_time : control_conf_->lat_controller_conf()
                                       .dy2heading_time_table()
                                       .dy2heading_time_info()) {
    data_near_point_time_.push_back(std::make_pair(
        info_near_point_time.currt_dy(), info_near_point_time.value()));
  }
  for (auto info_far_point_time : control_conf_->lat_controller_conf()
                                      .dy2_heading_time_coff_table()
                                      .dy2_heading_time_coff_info()) {
    data_far_point_time_.push_back(std::make_pair(
        info_far_point_time.curvature(), info_far_point_time.value()));
  }
  for (auto info_max_steer_angle : control_conf_->lat_controller_conf()
                                       .angle_req_max_vlu_table()
                                       .angle_req_max_vlu_info()) {
    data_max_steer_angle_.push_back(std::make_pair(
        info_max_steer_angle.curvature(), info_max_steer_angle.value()));
  }
  for (auto info_slip_param : control_conf_->lat_controller_conf()
                                  .pid_p_param_table()
                                  .pid_p_param_info()) {
    data_slip_param_.push_back(
        std::make_pair(info_slip_param.speedkph(), info_slip_param.value()));
  }

  return true;
}

void LatControllerTruck::Stop() {}

bool LatControllerTruck::Reset() { return true; }

std::string LatControllerTruck::Name() const { return name_; }

bool LatControllerTruck::Init(
    std::shared_ptr<DependencyInjector> injector,
    const rina::control::ControlParaConf *control_conf) {
  control_conf_ = control_conf;

  ts_ = control_conf_->lat_controller_conf().ts();
  if (ts_ <= 0.0) {
    AERROR << "[LateralController] Invalid control update interval.";
  }

  max_theta_deg_table_ = std::make_unique<rina::control::Lookup1D>();
  prew_time_dt_theta_table_ = std::make_unique<rina::control::Lookup1D>();
  prev_time_dist_table_ = std::make_unique<rina::control::Lookup1D>();
  reach_time_theta_table_ = std::make_unique<rina::control::Lookup1D>();
  max_steer_angle_table_ = std::make_unique<rina::control::Lookup1D>();
  near_point_time_table_ = std::make_unique<rina::control::Lookup1D>();
  far_point_time_table_ = std::make_unique<rina::control::Lookup1D>();
  slip_param_table_ = std::make_unique<rina::control::Lookup1D>();

  wheelbase_ = vehicle_param_.wheel_base;
  steer_ratio_ = vehicle_param_.steer_ratio;
  steer_single_direction_max_degree_ = vehicle_param_.max_steer_angle;

  // query_relative_time_ = control_conf_->query_relative_time();

  minimum_speed_protection_ = control_conf_->minimum_speed_protection();

  if (!LoadControlConf(control_conf_)) {
    AERROR << "failed to load control conf";
    return false;
  }
  return true;
};

bool LatControllerTruck::ComputeControlCommand(
    const rina::localization::GlobalPose *localization,
    const udp::VehicleInfoToADU *chassis_can,
    const rina::planning::ADCTrajectory *trajectory, const bool function_enable,
    rina::control::ControlCommand *cmd) {
  chassis_can_ = chassis_can;
  double steering_position = 0.0;

  ctrl_enable_ = function_enable;
  current_speed_mps_ = std::max(
      chassis_can_->vehicleinfoadata().ad18fef100_ccvs_vehspd() / kMps2Kph,
      static_cast<double>(minimum_speed_protection_));
  current_speed_kph_ = current_speed_mps_ * kMps2Kph;
  rina::planning::ChangeLaneType changeLaneType =
      trajectory->decision().main_decision().cruise().change_lane_type();
  AINFO << "planner_turnlight : " << changeLaneType;
  auto target_tracking_trajectory = *trajectory;
  trajectory_analyzer_ =
      std::move(TrajectoryAnalyzer(&target_tracking_trajectory));

  SimpleLateralDebug *debug = cmd->mutable_debug()->mutable_simple_lat_debug();
  debug->Clear();

  driving_orientation_ = localization->euler_angles().z() * M_PI / kPideg;

  static bool ctrl_enable_pre = false;
  bool ctrl_first_active = false;

  AINFO << "ctrl_enable_ : " << ctrl_enable_
        << " ctrl_enable_pre : " << ctrl_enable_pre;
  if (ctrl_enable_ && !ctrl_enable_pre) {
    ctrl_first_active = true;
    last1step_angle_ =
        chassis_can_->vehicleinfoadata().ad18fe313_steering_angle();
  }
  ctrl_enable_pre = ctrl_enable_;

  UpdateState(debug, localization);

  double target_steering_angle_deg = 0.;
  if (ctrl_enable_) {
    double yawrate_rads =
      chassis_can_->vehicleinfoaddata().ad18f0090b_vdc2_yawrate();
    double veh_spd_mps =
      chassis_can_->vehicleinfoaddata().ad18ef100_ccvs_vehspd() / kMps2Kph;
    double max_theta_deg = max_theta_deg_table_->Interpolate(
      data_max_theta_deg_.size(), data_max_theta_deg_,
      std::fabs(current_speed_kph_));
    double prew_time_dt_theta = prew_time_dt_theta_table_->Interpolate(
      data_prew_time_dt_theta_.size(), data_prew_time_dt_theta_,
      std::fabs(current_speed_kph_));
    double prev_time_dist = prev_time_dist_table_->Interpolate(
      data_prev_time_dist_.size(), data_prev_time_dist_,
      std::fabs(current_speed_kph_));
    double reach_time_theta = reach_time_theta_table_->Interpolate(
      data_reach_time_theta_.size(), data_reach_time_theta_,
      std::fabs(current_speed_kph_));
    near_point_time_ = near_point_time_table_->Interpolate(
      data_near_point_time_.size(), data_near_point_time_,
      std::fabs(current_speed_kph_));
    far_point_time_ = far_point_time_table_->Interpolate(
      data_far_point_time_.size(), data_far_point_time_,
      std::fabs(current_speed_kph_));
    double max_steer_angle = max_steer_angle_table_->Interpolate(
      data_max_steer_angle_.size(), data_max_steer_angle_,
      std::fabs(current_speed_kph_));
    slip_param_ = slip_param_table_->Interpolate(data_slip_param_.size(),
      data_slip_param_,
      std::fabs(current_speed_kph_));

    double real_theta =
      calculateRealTheta(-heading_error_, kLh, yawrate_rads, veh_spd_mps);
    AINFO << "real_theta :" << real_theta;
    double real_dt_theta =
      calculateRealDtTheta(yawrate_rads, curvature_far_, veh_spd_mps);
    AINFO << "real_dt_theta :" << real_dt_theta;

    double target_theta = 0.;
    double target_dt_theta = 0.;
    calculateTargetTheta(real_theta, -lateral_error_currt_, veh_spd_mps,
      prev_time_dist, kMin_prev_dist, max_theta_deg,
      &target_theta, &target_dt_theta);
    AINFO << "target_theta :" << target_theta
      << ", target_dt_theta :" << target_dt_theta;

    double target_curvature = calculateTargetCurvature(
      real_theta, target_theta, target_dt_theta, real_dt_theta, veh_spd_mps,
      reach_time_theta, prew_time_dt_theta, kMin_reach_dis_theta);
    AINFO << "target_curvature :" << target_curvature;

    double steer_angle_fb_raw =
      calculateSteeringAngle(target_curvature, wheelbase_, kLh, steer_ratio_);
    double filter_steer_angle_fb = rateLimitFilter(
      steer_angle_fb_raw_pre_, steer_angle_fb_raw, kRate_limit_fb, ts_);
    steer_angle_fb_raw_pre_ = filter_steer_angle_fb;

    double steer_angle_ff_raw =
      calculateSteeringAngle(target_curvature, curvature_far_, wheelbase_, kLh, steer_ratio_);
    double filter_steer_angle_ff = rateLimitFilter(
      steer_angle_ff_raw_pre_, steer_angle_ff_raw, kRate_limit_ff, ts_);
    steer_angle_ff_raw_pre_ = filter_steer_angle_ff;

    AINFO << "FB RAW:" << steer_angle_fb_raw
      << ", FF RAW:" << steer_angle_ff_raw;
    AINFO << "FB filter:" << filter_steer_angle_fb
      << ", FF filter:" << filter_steer_angle_ff;

    double target_steering_angle_raw =
      std::clamp(filter_steer_angle_fb + filter_steer_angle_ff,
        -max_steer_angle, max_steer_angle);

    double ratelimit_target_steering_angle = rateLimitFilter(
      laststep_angle_, target_steering_angle_raw, kRate_limit_total, ts_);

    target_steering_angle_deg = ratelimit_target_steering_angle;
    AINFO << " laststep_angle_ : " << laststep_angle_ << ",";
    laststep_angle_ = target_steering_angle_deg;
    AINFO << " real_angle_ : "
      << chassis_can_->vehicleinfoaddata().ad18ffe313_steering_angle()
      << ",";
    AINFO << " ffb_steering_angle : "
      << filter_steer_angle_fb + filter_steer_angle_ff << ",";
    AINFO << " ratelimit_target_steering_angle : "
      << ratelimit_target_steering_angle << ",";
    AINFO << " target_steering_angle_raw : " << target_steering_angle_raw << ",";

    // turnlight control
    if (changeLaneType == rina::planning::ChangeLaneType::LEFT) {
      cmd_->set_turnsignal(rina::control::TurnSignal::TURN_LEFT);
    } else {
      if (changeLaneType == rina::planning::ChangeLaneType::RIGHT) {
        cmd_->set_turnsignal(rina::control::TurnSignal::TURN_RIGHT);
      } else {
        cmd_->set_turnsignal(rina::control::TurnSignal::TURN_NONE);
      }
    }
  } else {
    target_steering_angle_deg = 0.;
    steer_angle_fb_raw_pre_ = 0.;
    steer_angle_ff_raw_pre_ = 0.;
    laststep_angle_ =
      chassis_can_->vehicleinfoaddata().ad18ffe313_steering_angle();
    cmd_->set_turnsignal(rina::control::TurnSignal::TURN_NONE);
  }
  AINFO << " target_steering_angle_deg :" << target_steering_angle_deg;
  cmd_->set_steering_target(target_steering_angle_deg);
  cmd_->set_path_curvature_current(current_curvature_);
  cmd_->set_path_curvature_near(curvature_near_);
  cmd_->set_path_curvature_far(curvature_far_);

  // debug->set_steer_mrac_enable_status(use_far_point_as_ff_);
  // debug->set_final_steer1(steer_angle_req_raw_);
  // debug->set_steer_angle_feedforward(turn_angle_req_ff_);
  // debug->set_steer_angle_feedback(turn_angle_req_fb_);

  // debug->set_heading(driving_orientation_);
  // debug->set_steer_angle(steer_angle_);
  // debug->set_steering_position(steering_position_);
  // debug->set_ref_speed(
  //   chassis_can_->vehicleinfoaddata().ad18ef100_ccvs_vehspd() /
  //   kMps2Kph);

  return true;
}

void LatControllerTruck::UpdateState(
  SimpleLateralDebug *debug,
  const rina::localization::GlobalPose *localization) {
  ComputeLateralErrors(
    localization->position_enu().x(), localization->position_enu().y(),
    driving_orientation_,
    chassis_can_->vehicleinfoaddata().ad18ef100_ccvs_vehspd() / kMps2Kph,  // ?? screenshot 10 line 294 reads as `ad18fef100` but same field elsewhere (lines 11, 126) reads `ad18ef100`; harmonized to `ad18ef100` [10.png]
    chassis_can_->vehicleinfoaddata().ad18f0090b_vdc2_yawrate(),
    chassis_can_->vehicleinfoaddata()
      .ad18f0090b_vdc2_longitudinalacceleration(),
    trajectory_analyzer_, debug);
}

void LatControllerTruck::ComputeLateralErrors(
  const double x, const double y, const double theta, const double linear_v,
  const double angular_v, const double linear_a,
  const TrajectoryAnalyzer &trajectory_analyzer, SimpleLateralDebug *debug) {
  rina::common::TrajectoryPoint target_point_currt;
  rina::common::TrajectoryPoint target_point_near;
  rina::common::TrajectoryPoint target_point_far;
  rina::common::TrajectoryPoint target_point_ff;
  if (trajectory_analyzer.trajectory_points().empty()) return;
  target_point_currt = trajectory_analyzer.QueryNearestPointByPosition(x, y);
  target_point_near = trajectory_analyzer.QueryNearestPointByRelativeTime(
    target_point_currt.relative_time() + near_point_time_);
  target_point_far = trajectory_analyzer.QueryNearestPointByRelativeTime(
    target_point_currt.relative_time() + far_point_time_);
  target_point_ff = trajectory_analyzer_.QueryNearestPointByRelativeTime(
      target_point_curt.relative_time_);

  const double dx = x_ - target_point_curt.path_point().x();
  const double dy = y_ - target_point_curt.path_point().y();

  const double cos_target_heading =
      std::cos(target_point_curt.path_point().theta());
  const double sin_target_heading =
      std::sin(target_point_curt.path_point().theta());

  AINFO << "curt point: " << target_point_curt.path_point().x() << " | "
        << target_point_curt.path_point().y();
  AINFO << "far point: " << target_point_far.path_point().x() << " | "
        << target_point_far.path_point().y();

  lateral_error_curt_ = cos_target_heading * dy - sin_target_heading * dx;
  heading_error_ = ring::common::math::NormalizeAngle(  // ?? ring 命名空间，可能为 mlp::common::math [11.png]
      angle_ - target_point_curt.path_point().theta());
  AINFO << "wv: " << lateral_error_curt_ << ", theta error: " << heading_error_;  // ?? "wv" 字符串可能是 "e_lat" 等 [11.png]

  debug->set_lateral_error(lateral_error_curt_);
  debug->set_ref_heading(target_point_near.path_point().theta());
  debug->set_ref_heading_far(target_point_far.path_point().theta());
  // debug->set_heading_error_near(heading_error_near_);
  // debug->set_heading_error_far(heading_error_far_);
  debug->set_curvature(current_curvature_);
  debug->set_heading_error(heading_error_);

  current_curvature_raw_ = target_point_near.path_point().kappa();
  curvature_near_raw_ = target_point_near.path_point().kappa();
  curvature_far_raw_ = target_point_far.path_point().kappa();

  current_curvature_ =
      (current_curvature_raw_ * 1.0) + ((1 - 1.0) * current_curvature_pre_);
  current_curvature_pre_ = current_curvature_;

  curvature_near_ =
      (curvature_near_raw_ * 1.0) + ((1 - 1.0) * curvature_near_pre_);
  curvature_near_pre_ = curvature_near_;

  curvature_far_ =
      (curvature_far_raw_ * 1.0) + ((1 - 1.0) * curvature_far_pre_);
  curvature_far_pre_ = curvature_far_;
  curvature_ff_ = target_point_ff.path_point().kappa();

  AINFO << "curvature_current = " << current_curvature_
        << ", curvature_near = " << curvature_near_
        << ", curvature_far = " << curvature_far_;
}

// theta HA偏差
// wheel_base_lh 默认 0
double LatControllerTruck::calculateRealTheta(double theta,
                                              double wheel_base_lh,
                                              double yawrate_rads,
                                              double vehspd_mps) {
  double vehicle_speed = std::clamp(vehspd_mps, kMinSpeed, kMaxSpeed);
  double intermediate = (wheel_base_lh * yawrate_rads) / vehicle_speed;
  double result = theta - std::atan(intermediate);
  return result;
}

// theta2lane = real theta
// dis2lane e
void LatControllerTruck::calculateTargetTheta(
    double theta2lane, double dis2lane, double vehicle_speed_mps,
    double prev_time_dist, double min_prev_dist, double max_theta_deg,
    double &target_theta, double &target_dt_theta) {
  double prev_dist =
      std::max(vehicle_speed_mps * prev_time_dist, min_prev_dist);

  double error_angle_raw = std::atan(dis2lane / prev_dist);

  double max_error_angle =
      std::min(max_theta_deg * M_PI / kPideg, std::abs(error_angle_raw));

  double sign_error_angle_raw = 0.;
  if (std::abs(error_angle_raw) < 1e-3) {
    sign_error_angle_raw = 0.;
  } else {
    sign_error_angle_raw = error_angle_raw < 0. ? -1.0 : 1.0;
  }

  target_theta = -1.0 * sign_error_angle_raw * max_error_angle;

  target_dt_theta = std::sin(theta2lane) * vehicle_speed_mps * prev_dist
                    / (prev_dist * prev_dist + dis2lane * dis2lane) * -1.0;
}

double LatControllerTruck::calculateRealDtTheta(double yawrate_rads,
                                                double road_curvature,
                                                double veh_spd_mps) {
  return (yawrate_rads - road_curvature * veh_spd_mps) * -1.0;
}

double LatControllerTruck::calculateTargetCurvature(
    double real_theta, double target_theta, double target_dt_theta,
    double real_dt_theta, double veh_spd_mps, double reach_time_theta,
    double prew_time_dt_theta, double min_reach_dis_theta) {
  double target_curvature =
      -1.0
      * ((target_theta - real_theta)
         + (target_dt_theta - real_dt_theta) * prew_time_dt_theta)
      / std::max(reach_time_theta * veh_spd_mps, min_reach_dis_theta);
  return target_curvature;
}

double LatControllerTruck::calculateSteeringAngle(double target_curvature,
                                                  double wheel_base, double lh,
                                                  double eps_gear_ratio) {
  double front_axis_sa = std::atan(
      target_curvature * wheel_base
      / std::sqrt(1.0 - target_curvature * lh * target_curvature * lh));
  double front_axis_sa_deg = front_axis_sa * M_PI / kPideg;
  double steering_angle = front_axis_sa_deg * eps_gear_ratio * slip_param_;

  return steering_angle;
}

double LatControllerTruck::rateLimitFilter(double current_angle,
                                           double target_angle,
                                           double rate_limit,
                                           double delta_time) {
  double max_delta_angle = rate_limit * delta_time;

  double delta_angle = target_angle - current_angle;

  if (delta_angle > max_delta_angle) {
    delta_angle = max_delta_angle;
  } else if (delta_angle < -max_delta_angle) {
    delta_angle = -max_delta_angle;
  }

  return current_angle + delta_angle;
}

}  // namespace control
}  // namespace mlp
