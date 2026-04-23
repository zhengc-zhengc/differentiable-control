"""对比 sim_loop 合成的 yawrate 和 plant 真实 yawrate 的差距（truck_trailer 圆弧）。"""
import sys
import math
sys.path.insert(0, '.')
import torch
from config import load_config
from model.trajectory import generate_circle
from model.vehicle_factory import create_vehicle, resolve_vehicle_geometry
from model.trajectory import TrajectoryAnalyzer
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from common import normalize_angle

DEG2RAD = math.pi / 180.0

cfg = load_config('configs/default.yaml')
cfg['vehicle']['model_type'] = 'truck_trailer'

traj = generate_circle(radius=30.0, speed=5.0,
                       arc_angle=math.pi * 1.5, dt=0.02)
analyzer = TrajectoryAnalyzer(traj)
wheelbase, steer_ratio = resolve_vehicle_geometry(cfg)
dt = 0.02

car = create_vehicle(cfg, x=traj[0].x, y=traj[0].y, yaw=traj[0].theta,
                     v=5.0, dt=dt, differentiable=False)
lat = LatControllerTruck(cfg, differentiable=False)
lon = LonController(cfg, differentiable=False)

n_steps = len(traj)
prev_steer = 0.0
v_prev = 5.0
ref_kappa = traj[0].kappa

print(f"{'t':>5} {'yawrate_synth':>13} {'yawrate_plant':>13} "
      f"{'κ_ref·v':>9} {'steer_prev':>10} {'dt_theta_err_synth':>18} "
      f"{'dt_theta_err_real':>17}")

for step in range(n_steps):
    t = step * dt
    car_x = car.x.item(); car_y = car.y.item()
    car_v = car.v.item(); car_yaw = car.yaw.item()
    car_speed_kph = car.speed_kph.item()

    delta_prev = prev_steer / steer_ratio * DEG2RAD
    yawrate_synth = car_v * math.tan(delta_prev) / wheelbase

    # 真实 plant yaw rate: 牵引车 r_s 分量在 TruckTrailerVehicle 内部
    # 通过 get_state_vector 取：顺序参考 model/truck_trailer_vehicle.py
    # state[:, 7] 是 r_s（牵引车横摆角速度）
    state = car._state if hasattr(car, '_state') else None
    # state[5] = r_t (tractor yaw rate) per truck_trailer_dynamics
    if state is not None and state.numel() >= 6:
        yawrate_plant = float(state[5].item())
    else:
        yawrate_plant = float('nan')

    kappa_v = ref_kappa * car_v
    dt_theta_err_synth = yawrate_synth - kappa_v
    dt_theta_err_real = yawrate_plant - kappa_v

    if step % 50 == 0:
        print(f"{t:5.2f} {yawrate_synth:13.5f} {yawrate_plant:13.5f} "
              f"{kappa_v:9.4f} {prev_steer:10.2f} "
              f"{dt_theta_err_synth:18.5f} {dt_theta_err_real:17.5f}")

    steer_out, _, _, curv_far, _, _ = lat.compute(
        x=car_x, y=car_y, yaw_deg=math.degrees(car_yaw),
        speed_kph=car_speed_kph, yawrate=yawrate_synth,
        steer_feedback=prev_steer, analyzer=analyzer,
        ctrl_enable=True, dt=dt)
    acc_cmd = lon.compute(
        x=car_x, y=car_y, yaw_deg=math.degrees(car_yaw),
        speed_kph=car_speed_kph, accel_mps2=0.0,
        curvature_far=curv_far, analyzer=analyzer, t_now=t,
        ctrl_enable=True, ctrl_first_active=(step == 0), dt=dt)
    delta_front = steer_out / steer_ratio * DEG2RAD
    a_actual = (car_v - v_prev) / dt
    tq = lon.compute_torque_wheel(acc_cmd, car_v, a_actual)
    v_prev = car_v
    car.step(delta=delta_front, torque_wheel=tq)
    prev_steer = steer_out
