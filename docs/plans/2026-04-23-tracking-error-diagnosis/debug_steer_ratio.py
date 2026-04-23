"""查传动比：controller 用的 vs plant 用的是否一致。
逐层打印：config → controller.steer_ratio → sim_loop 除的 ratio → adapter 乘的 ratio → dynamics 除的 ratio。
如果任何一处不一致，end-to-end 的 δ_front 就会错一个系数。
"""
import sys
import math
sys.path.insert(0, '..')
sys.path.insert(0, '../../../.worktrees/debug-tracking-error/sim')

# 加这一条确保能找到 sim 目录
import os
HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'sim'))
sys.path.insert(0, SIM_DIR)

from config import load_config
from model.vehicle_factory import create_vehicle, resolve_vehicle_geometry
from controller.lat_truck import LatControllerTruck


for plant in ['kinematic', 'dynamic', 'hybrid_dynamic', 'truck_trailer']:
    print(f"\n=== plant = {plant} ===")
    cfg = load_config(os.path.join(SIM_DIR, 'configs', 'default.yaml'))
    cfg['vehicle']['model_type'] = plant

    # 1. config 里配的
    if plant == 'kinematic':
        cfg_ratio = cfg['vehicle']['steer_ratio']
    elif plant == 'dynamic':
        cfg_ratio = cfg['dynamic_vehicle']['steer_ratio']
    elif plant == 'hybrid_dynamic':
        cfg_ratio = cfg['hybrid_dynamic_vehicle']['steer_ratio']
    elif plant == 'truck_trailer':
        cfg_ratio = cfg['truck_trailer_vehicle']['steering_ratio']
    print(f"  [config]        steer_ratio = {cfg_ratio}")

    # 2. resolve_vehicle_geometry → controller & sim_loop 都用这个
    L, r = resolve_vehicle_geometry(cfg)
    print(f"  [resolve]       steer_ratio = {r}  (wheelbase={L})")

    # 3. controller 内部持有的
    lat = LatControllerTruck(cfg, differentiable=False)
    print(f"  [controller]    self.steer_ratio = {lat.steer_ratio}")

    # 4. plant adapter 内部持有的
    try:
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=0, dt=0.02, differentiable=False)
    except Exception as e:
        print(f"  [plant]         create failed: {e}")
        continue
    adapter_ratio = getattr(car, '_steer_ratio', None)
    print(f"  [plant adapter] self._steer_ratio = {adapter_ratio}")

    # 5. plant dynamics 内部持有的
    dyn = getattr(car, 'dynamics', None)
    if dyn is not None:
        dyn_ratio = getattr(dyn, 'steer_ratio', None) or getattr(dyn, 'steering_ratio', None)
        if hasattr(dyn_ratio, 'item'):
            dyn_ratio = dyn_ratio.item()
        print(f"  [dynamics]      steer_ratio/steering_ratio = {dyn_ratio}")
    else:
        print(f"  [dynamics]      (kinematic plant, no separate dynamics layer)")

    # 一致性检查
    all_ratios = [cfg_ratio, r, lat.steer_ratio]
    if adapter_ratio is not None:
        all_ratios.append(adapter_ratio)
    if dyn is not None and dyn_ratio is not None:
        all_ratios.append(dyn_ratio)
    if all(abs(x - all_ratios[0]) < 1e-9 for x in all_ratios):
        print(f"  >> 所有层 ratio 一致：{all_ratios[0]}")
    else:
        print(f"  >> 不一致！各层 ratio = {all_ratios}")

    # 端到端验证：controller 输出 100 deg，经过所有层，最终到 dynamics 的 delta_f 是多少？
    steer_out_deg = 100.0
    # sim_loop: delta_front = steer_out / r * DEG2RAD
    delta_front_sim_loop = steer_out_deg / r * (math.pi / 180)
    print(f"  e2e: steer_out={steer_out_deg}deg → sim_loop gives δ_front = {math.degrees(delta_front_sim_loop):.4f} deg (rad={delta_front_sim_loop:.5f})")
    # adapter: delta_sw = delta * adapter_ratio
    # dynamics: delta_f = delta_sw / dyn_ratio
    # 所以 final δ_f = delta_front_sim_loop * (adapter_ratio / dyn_ratio)
    if adapter_ratio is not None and dyn is not None and dyn_ratio is not None:
        final_delta_f = delta_front_sim_loop * adapter_ratio / dyn_ratio
        print(f"       adapter×{adapter_ratio} / dynamics÷{dyn_ratio} → 最终 δ_f = {math.degrees(final_delta_f):.4f} deg")
        if abs(final_delta_f - delta_front_sim_loop) < 1e-9:
            print(f"       [OK] adapter 和 dynamics 的 ratio 对消，端到端正确")
        else:
            print(f"       [BAD] 不对消！端到端误差 = {math.degrees(final_delta_f - delta_front_sim_loop):.4f} deg")
