"""实验 G/H：把 sim_loop 里的 "synth yawrate" 换成 plant 真实 yawrate，看能否改善 truck_trailer 圆弧。"""
import sys
import math
sys.path.insert(0, '.')
import torch
from config import load_config
from model.trajectory import generate_circle, TrajectoryAnalyzer
from model.vehicle_factory import create_vehicle, resolve_vehicle_geometry
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from common import normalize_angle

DEG2RAD = math.pi / 180.0


def run_custom(cfg, use_plant_yawrate: bool):
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
    lat_errs = []
    for step in range(n_steps):
        t = step * dt
        car_x = car.x.item(); car_y = car.y.item()
        car_v = car.v.item(); car_yaw = car.yaw.item()
        car_speed_kph = car.speed_kph.item()

        # yawrate: synth 或 plant 真值
        if use_plant_yawrate and hasattr(car, '_state') and car._state.numel() >= 6:
            yawrate = float(car._state[5].item())
        else:
            delta_prev = prev_steer / steer_ratio * DEG2RAD
            yawrate = car_v * math.tan(delta_prev) / wheelbase

        # lat err (just for tracking)
        ref_pt = analyzer.query_nearest_by_position(car_x, car_y)
        dx = car_x - ref_pt.x; dy = car_y - ref_pt.y
        lat_err = math.cos(ref_pt.theta) * dy - math.sin(ref_pt.theta) * dx
        lat_errs.append(lat_err)

        steer_out, _, _, curv_far, _, _ = lat.compute(
            x=car_x, y=car_y, yaw_deg=math.degrees(car_yaw),
            speed_kph=car_speed_kph, yawrate=yawrate,
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
    return lat_errs


def dump(name, errs):
    print(f"{name:55s}  |lat_err|max={max(abs(e) for e in errs):6.2f}m  "
          f"last={abs(errs[-1]):6.2f}m")


if __name__ == '__main__':
    cfg = load_config('configs/default.yaml')
    cfg['vehicle']['model_type'] = 'truck_trailer'

    dump('G1. truck_trailer + synth yawrate (current)', run_custom(cfg, False))
    dump('G2. truck_trailer + plant real yawrate', run_custom(cfg, True))

    # + 同时放开 T1 clamp
    cfg2 = load_config('configs/default.yaml')
    cfg2['vehicle']['model_type'] = 'truck_trailer'
    cfg2['lat_truck']['T1_max_theta_deg'] = [[s, 45.0]
                                             for s in [0, 10, 20, 30, 40, 50, 60]]
    dump('H1. + uncap T1 (synth yawrate)', run_custom(cfg2, False))
    dump('H2. + uncap T1 (plant real yawrate)', run_custom(cfg2, True))
