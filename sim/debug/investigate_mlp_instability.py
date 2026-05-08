"""调研 0507 MLP 闭环失控原因。

核心问题：加入 0507 MLP 后整车轨迹完全南辕北辙、出现折返。
究竟是某一步 MLP 输出离谱导致瞬时崩溃，还是误差累积过程？

策略：
1. 同一场景跑 4 个变体（无 MLP / 0507 完整 / 0507 仅留挂车残差 / 0507 仅留牵引车残差），
   闭环全程 forward hook 抓 MLP 14D 输入、归一化输入、9D 原始输出、9D 裁剪输出。
2. 同步保存控制器命令、车辆状态、跟踪误差、与"无 MLP 基准"的逐步状态偏差。
3. 数据落到 npz，由分析脚本绘图。

支持场景：lane_change_5kph / clothoid_left_5kph / circle_25kph_R80 / straight_5kph。
"""
import sys, os, copy, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import load_config, apply_plant_override
from model.trajectory import (generate_straight, generate_circle,
                              expand_trajectories)
from sim_loop import run_simulation
import model.truck_trailer_vehicle as ttv


# ===== 全局捕获缓冲 =====
_capture = {
    'mlp_input_raw': [],
    'mlp_input_norm': [],
    'mlp_output_raw': [],
    'mlp_output_clipped': [],
    'state_before_mlp': [],   # base_next, 12D
    'state_after_mlp': [],    # final state, 12D
    'control': [],            # 5D control [steer_sw, T_FL, T_FR, T_RL, T_RR]
}
_mask = None  # 9D mask applied to MLP output (None = identity)


def reset_capture():
    for k in _capture:
        _capture[k].clear()


def capture_arrays():
    return {k: (np.array(v) if len(v) > 0 else None)
            for k, v in _capture.items()}


# ===== Monkey-patch step() 以抓全套数据 =====
_orig_step = ttv.TruckTrailerVehicle.step


def _patched_step(self, delta, torque_wheel):
    """Reimplements step() while logging every intermediate. mask 可以裁剪 MLP 输出。"""
    if not isinstance(delta, torch.Tensor):
        delta = torch.tensor(float(delta))
    if not isinstance(torque_wheel, torch.Tensor):
        torque_wheel = torch.tensor(float(torque_wheel))

    delta_sw = delta * self._steer_ratio
    torque_rear = torque_wheel / 2.0
    zero = torch.zeros_like(torque_wheel)
    control = torch.stack(
        [delta_sw, zero, zero, torque_rear, torque_rear]).unsqueeze(0)

    state = self._state.unsqueeze(0)
    trailer_mass = state.new_tensor([[self._trailer_mass]])
    dt_t = state.new_tensor([[self.dt]])

    base_next = self.dynamics(state, control, trailer_mass, dt_t)

    if self._mlp is not None:
        if self._mlp_input_dim == 14:
            features = ttv.build_mlp_input_feature_tensor_v2(
                state, control, trailer_mass, dt_t)
        else:
            features = ttv.build_mlp_input_feature_tensor(
                state, control, trailer_mass, dt_t)
        feat_raw = features.detach().cpu().numpy()[0].copy()
        if self._feature_mean is not None:
            features_norm = (features - self._feature_mean) / self._feature_scale
        else:
            features_norm = features
        feat_norm = features_norm.detach().cpu().numpy()[0].copy()

        motion_error_raw = self._mlp(features_norm)
        motion_out_raw_np = motion_error_raw.detach().cpu().numpy()[0].copy()

        if self._motion_error_clip is not None:
            motion_error = torch.clamp(
                motion_error_raw,
                -self._motion_error_clip,
                self._motion_error_clip)
        else:
            motion_error = motion_error_raw

        # 应用 mask 做组件消融
        global _mask
        if _mask is not None:
            mask_t = motion_error.new_tensor(_mask).reshape(1, -1)
            motion_error = motion_error * mask_t

        motion_out_clip_np = motion_error.detach().cpu().numpy()[0].copy()

        if self._mlp_output_dim == 9:
            full_error = ttv.derive_full_error_from_motion_error_torch_v2(
                motion_error, base_next, dt_t)
        else:
            full_error = ttv.derive_full_error_from_motion_error_torch(
                motion_error, base_next, dt_t)
        new_state_full = (base_next + full_error).squeeze(0).clone()
        new_state_full[2] = ttv.wrap_angle_error_torch(
            new_state_full[2:3])[0]
        new_state_full[8] = ttv.wrap_angle_error_torch(
            new_state_full[8:9])[0]
        if self._trailer_mass <= ttv.NO_TRAILER_MASS_THRESHOLD_KG:
            new_state_full[6] = new_state_full[0]
            new_state_full[7] = new_state_full[1]
            new_state_full[8] = new_state_full[2]
            new_state_full[9] = new_state_full[3]
            new_state_full[10] = new_state_full[4]
            new_state_full[11] = new_state_full[5]
        self._state = new_state_full

        _capture['mlp_input_raw'].append(feat_raw)
        _capture['mlp_input_norm'].append(feat_norm)
        _capture['mlp_output_raw'].append(motion_out_raw_np)
        _capture['mlp_output_clipped'].append(motion_out_clip_np)
    else:
        _capture['mlp_input_raw'].append(np.zeros(14))
        _capture['mlp_input_norm'].append(np.zeros(14))
        _capture['mlp_output_raw'].append(np.zeros(9))
        _capture['mlp_output_clipped'].append(np.zeros(9))
        self._state = base_next.squeeze(0)

    _capture['state_before_mlp'].append(
        base_next.detach().cpu().numpy()[0].copy())
    _capture['state_after_mlp'].append(
        self._state.detach().cpu().numpy().copy())
    _capture['control'].append(control.detach().cpu().numpy()[0].copy())


ttv.TruckTrailerVehicle.step = _patched_step


# ===== 场景定义 =====
def make_scenarios(filter_keys=None):
    scenarios = []

    v5 = 5.0 / 3.6
    traj = generate_straight(length=30.0, speed=v5, dt=0.02)
    scenarios.append(('straight_5kph', '直行 5 kph (30 m)', traj, v5))

    v25 = 25.0 / 3.6
    traj = generate_circle(radius=80.0, speed=v25,
                           arc_angle=math.pi / 2, dt=0.02)
    scenarios.append(('circle_25kph_R80', '稳态圆周 25 kph (R=80 m)',
                      traj, v25))

    expanded = expand_trajectories(['lane_change'])
    key, label, gen = next(t for t in expanded if '5kph' in t[0])
    traj = gen()
    scenarios.append(('lane_change_5kph', '低速变道 5 kph', traj, traj[0].v))

    expanded = expand_trajectories(['clothoid_left'])
    key, label, gen = next(t for t in expanded if '5kph' in t[0])
    traj = gen()
    scenarios.append(('clothoid_left_5kph', 'clothoid 左转 5 kph',
                      traj, traj[0].v))

    if filter_keys:
        scenarios = [s for s in scenarios if s[0] in filter_keys]
    return scenarios


# ===== 配置加载 =====
def load_cfg_for_variant(variant):
    """variant: 'no_mlp' | 'mlp_default' | 'mlp_0507' """
    if variant == 'no_mlp':
        cfg = load_config('configs/train_with_0507.yaml')
        apply_plant_override(cfg, 'truck_trailer')
        cfg['truck_trailer_vehicle']['checkpoint_path'] = ''
    elif variant == 'mlp_default':
        cfg = load_config()
        apply_plant_override(cfg, 'truck_trailer')
    elif variant == 'mlp_0507':
        cfg = load_config('configs/train_with_0507.yaml')
        apply_plant_override(cfg, 'truck_trailer')
    else:
        raise ValueError(variant)
    return cfg


# ===== 跑一个 (场景, 变体) =====
def run_one(scenario_key, label, traj, init_v, variant, mask=None):
    global _mask
    _mask = mask
    cfg = load_cfg_for_variant(variant)
    reset_capture()

    t0 = time.time()
    history = run_simulation(traj, init_speed=init_v, cfg=cfg)
    dt_run = time.time() - t0

    arrays = capture_arrays()
    arrays['variant'] = variant
    arrays['mask'] = mask if mask is not None else np.ones(9)
    arrays['ckpt_path'] = cfg['truck_trailer_vehicle'].get(
        'checkpoint_path', '')
    arrays['scenario_key'] = scenario_key
    arrays['scenario_label'] = label

    # 历史展开成 numpy
    arrays['hist_t'] = np.array([float(h['t']) for h in history])
    arrays['hist_x'] = np.array([float(h['x']) for h in history])
    arrays['hist_y'] = np.array([float(h['y']) for h in history])
    arrays['hist_yaw'] = np.array([float(h['yaw']) for h in history])
    arrays['hist_v'] = np.array([float(h['v']) for h in history])
    arrays['hist_steer'] = np.array([float(h['steer']) for h in history])
    arrays['hist_steer_fb'] = np.array([float(h['steer_fb']) for h in history])
    arrays['hist_steer_ff'] = np.array([float(h['steer_ff']) for h in history])
    arrays['hist_acc'] = np.array([float(h['acc']) for h in history])
    arrays['hist_lat_err'] = np.array(
        [float(h['lateral_error']) for h in history])
    arrays['hist_head_err'] = np.array(
        [float(h['heading_error']) for h in history])
    arrays['hist_ref_x'] = np.array([float(h['ref_x']) for h in history])
    arrays['hist_ref_y'] = np.array([float(h['ref_y']) for h in history])

    # 参考轨迹 (xy + theta + v)
    arrays['ref_traj_x'] = np.array([p.x for p in traj])
    arrays['ref_traj_y'] = np.array([p.y for p in traj])
    arrays['ref_traj_theta'] = np.array([p.theta for p in traj])
    arrays['ref_traj_v'] = np.array([p.v for p in traj])

    print(f"  {variant:<22} | t={dt_run:5.1f}s | "
          f"lat_RMSE={np.sqrt(np.mean(arrays['hist_lat_err']**2)):.3f} | "
          f"|lat|max={np.max(np.abs(arrays['hist_lat_err'])):.3f} | "
          f"yaw_max={np.max(np.abs(arrays['hist_yaw'])):.2f}rad")
    return arrays


# ===== 主流程：每个场景跑 4 个变体并保存 =====
def run_scenario(scenario_key, label, traj, init_v, output_dir):
    print(f"\n=== 场景：{label} (n_pts={len(traj)}, v0={init_v:.2f} m/s) ===")
    scen_dir = os.path.join(output_dir, scenario_key)
    os.makedirs(scen_dir, exist_ok=True)

    variants = [
        ('no_mlp', None),
        ('mlp_default', None),
        ('mlp_0507_full', None),  # 0507 完整
        ('mlp_0507_zero_vel_t',  # 牵引车速度残差 (idx 0,1,2) 置零
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])),
        ('mlp_0507_zero_pose_s',  # 相对位姿残差 (idx 6,7,8) 置零
            np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])),
        ('mlp_0507_only_yaw_t',  # 仅留牵引车 yaw rate 残差 (idx 2)
            np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])),
        ('mlp_0507_only_vx_t',  # 仅留牵引车 vx 残差 (idx 0)
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('mlp_0507_only_vy_t',  # 仅留牵引车 vy 残差 (idx 1)
            np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])),
    ]

    all_results = {}
    for vname, mask in variants:
        # 0507 系列变体都用 train_with_0507.yaml + 适当的 mask
        if vname.startswith('mlp_0507'):
            arr = run_one(scenario_key, label, traj, init_v,
                          'mlp_0507', mask=mask)
        else:
            arr = run_one(scenario_key, label, traj, init_v, vname,
                          mask=mask)
        arr['variant'] = vname
        all_results[vname] = arr
        np.savez(os.path.join(scen_dir, f'{vname}.npz'), **arr)

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+', default=None,
                        help='过滤场景：lane_change_5kph clothoid_left_5kph 等')
    parser.add_argument('--out', default=None,
                        help='输出目录')
    args = parser.parse_args()

    output_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'diagnostic', 'mlp_instability')
    os.makedirs(output_dir, exist_ok=True)
    print(f"产物目录：{output_dir}")

    scenarios = make_scenarios(filter_keys=args.scenarios)
    for key, label, traj, v0 in scenarios:
        run_scenario(key, label, traj, v0, output_dir)

    print(f"\n全部数据已存：{output_dir}")


if __name__ == '__main__':
    main()
