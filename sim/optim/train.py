# sim/optim/train.py
"""V2 可微调参训练 pipeline。

DiffControllerParams 封装横向+纵向控制器，所有可优化参数通过 .parameters() 暴露。
tracking_loss 计算多项跟踪误差 + 平滑度惩罚。
train() 运行多轨迹多 epoch 梯度优化。

用法：
    python train.py --epochs 5 --trajectories circle --speed 5.0
"""
import argparse
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config, save_tuned_config, apply_plant_override
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from model.trajectory import (expand_trajectories, TRAJECTORY_TYPES,
                              SPEED_BANDS_KPH)
from sim_loop import run_simulation


class DiffControllerParams(nn.Module):
    """封装横向 + 纵向控制器。所有可优化参数通过 .parameters() 暴露。"""

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        self.lon_ctrl = LonController(cfg, differentiable=True)

    def to_config_dict(self):
        """将当前参数导出为 YAML 兼容的 dict（结构与 default.yaml 一致）。"""
        cfg = load_config()  # 模板

        # --- 横向控制器参数 ---
        lat = cfg['lat_truck']
        lat['kLh'] = float(self.lat_ctrl.kLh.item())
        for i, key in enumerate(['T1_max_theta_deg', 'T2_prev_time_dist',
                                  'T3_reach_time_theta', 'T4_T_dt',
                                  'T5_near_point_time', 'T6_far_point_time',
                                  'T7_max_steer_angle', 'T8_slip_param']):
            tx = getattr(self.lat_ctrl, f'T{i+1}_x')
            ty = getattr(self.lat_ctrl, f'T{i+1}_y')
            lat[key] = [[float(tx[j].detach()), float(ty[j].detach())]
                        for j in range(len(tx))]

        # --- 纵向控制器参数 ---
        lon = cfg['lon']
        lon['station_kp'] = float(self.lon_ctrl.station_kp.item())
        lon['station_ki'] = float(self.lon_ctrl.station_ki.item())
        lon['low_speed_kp'] = float(self.lon_ctrl.low_speed_kp.item())
        lon['low_speed_ki'] = float(self.lon_ctrl.low_speed_ki.item())
        lon['high_speed_kp'] = float(self.lon_ctrl.high_speed_kp.item())
        lon['high_speed_ki'] = float(self.lon_ctrl.high_speed_ki.item())
        lon['switch_speed'] = float(self.lon_ctrl.switch_speed.item())
        for i, key in enumerate(['L1_acc_up_lim', 'L2_acc_low_lim',
                                  'L3_acc_up_rate', 'L4_acc_down_rate',
                                  'L5_rate_gain']):
            tx = getattr(self.lon_ctrl, f'L{i+1}_x')
            ty = getattr(self.lon_ctrl, f'L{i+1}_y')
            lon[key] = [[float(tx[j].detach()), float(ty[j].detach())]
                        for j in range(len(tx))]

        return cfg


def tracking_loss(history, ref_speed,
                  w_lat=10.0, w_head=8.0, w_speed=3.0,
                  w_steer_rate=0.05, w_acc_rate=0.01,
                  return_details=False):
    """计算跟踪 loss：横向误差 + 航向误差 + 速度误差 + 平滑度惩罚。

    Args:
        history: run_simulation 返回的 differentiable=True 历史记录
        ref_speed: 参考速度 (m/s)
        w_lat/w_head/w_speed: 各误差项权重
        w_steer_rate/w_acc_rate: 平滑度惩罚权重
        return_details: 是否同时返回各分项指标（不含权重的原始值）

    Returns:
        loss (torch.Tensor): 标量 loss，支持 .backward()
        details (dict): 仅当 return_details=True 时返回，各分项原始指标
    """
    lat_errs = torch.stack([h['lateral_error'] for h in history])
    head_errs = torch.stack([h['heading_error'] for h in history])
    speeds = torch.stack([h['v'] for h in history])
    steers = torch.stack([h['steer'] for h in history])
    accs = torch.stack([h['acc'] for h in history])

    speed_errs = speeds - ref_speed

    lat_mse = (lat_errs ** 2).mean()
    head_mse = (head_errs ** 2).mean()
    speed_mse = (speed_errs ** 2).mean()

    loss = w_lat * lat_mse + w_head * head_mse + w_speed * speed_mse

    steer_rate_mse = torch.tensor(0.0)
    acc_rate_mse = torch.tensor(0.0)
    if len(steers) > 1:
        steer_rate = steers[1:] - steers[:-1]
        acc_rate = accs[1:] - accs[:-1]
        steer_rate_mse = (steer_rate ** 2).mean()
        acc_rate_mse = (acc_rate ** 2).mean()
        loss = loss + w_steer_rate * steer_rate_mse
        loss = loss + w_acc_rate * acc_rate_mse

    if return_details:
        details = {
            'lat_rmse': lat_mse.sqrt().item(),
            'head_rmse': head_mse.sqrt().item(),
            'speed_rmse': speed_mse.sqrt().item(),
            'lat_max': lat_errs.abs().max().item(),
            'head_max': head_errs.abs().max().item(),
            'loss_lat': (w_lat * lat_mse).item(),
            'loss_head': (w_head * head_mse).item(),
            'loss_speed': (w_speed * speed_mse).item(),
            'loss_steer_rate': (w_steer_rate * steer_rate_mse).item(),
            'loss_acc_rate': (w_acc_rate * acc_rate_mse).item(),
        }
        return loss, details

    return loss


def _build_traj_list(type_names, verbose=True):
    """将 CLI 类型名展开为 [(key, generator), ...] 训练列表。

    Args:
        type_names: 轨迹类型名列表，None 则使用全部 TRAJECTORY_TYPES (8×6=48 条)

    Returns:
        list of (key, lazy_generator) tuples
    """
    expanded = expand_trajectories(type_names)
    if verbose:
        n_types = len(type_names) if type_names else len(TRAJECTORY_TYPES)
        n_bands = len(SPEED_BANDS_KPH)
        print(f"训练轨迹: {n_types} 类型 × {n_bands} 速度段 = {len(expanded)} 条")
        types_str = ', '.join(type_names) if type_names else '全部'
        print(f"  类型: {types_str}")
        print(f"  速度段: {SPEED_BANDS_KPH} kph")
    return [(key, gen) for key, _label, gen in expanded]


def train(trajectories=None, n_epochs=100, lr=5e-2, lr_tables=5e-2,
          sim_length=None, tbptt_k=150, grad_clip=10.0,
          param_snapshot_interval=10, verbose=True, plant=None,
          config_path=None, w_lat=10.0, w_head=8.0, w_speed=3.0,
          w_steer_rate=0.05, w_acc_rate=0.01):
    """运行可微调参训练。

    Args:
        trajectories: 轨迹类型名列表，如 ['lane_change', 'clothoid_decel']。
                      每个类型自动展开到全 6 速度段。None 则全量训练 (8×6=48 条)。
        n_epochs: 训练轮数
        lr: 主学习率（PID 增益等标量参数）
        lr_tables: 查找表 y 值学习率（通常低于主学习率）
        sim_length: 仿真距离限制 (m)，None 为全长
        tbptt_k: Truncated BPTT 窗口大小（步数），每 K 步截断梯度链
        grad_clip: 全局梯度范数裁剪阈值
        param_snapshot_interval: 参数快照打印间隔（epoch 数），0 表示不打印
        plant: 被控对象类型 ('kinematic'/'dynamic')，None 使用配置默认值
        config_path: 初始参数配置路径，用于 warm-start（从上次调参结果继续）
        w_lat/w_head/w_speed: Loss 权重（横向/航向/速度误差）
        w_steer_rate/w_acc_rate: 平滑度惩罚权重
        verbose: 是否打印 epoch 信息

    Returns:
        dict: {'losses', 'training_history', 'initial_params', 'final_params',
               'saved_path', 'params'}
    """
    traj_list = _build_traj_list(trajectories, verbose=verbose)

    cfg = load_config(config_path)
    if plant:
        apply_plant_override(cfg, plant)
    params = DiffControllerParams(cfg=cfg)

    # 注册梯度钩子：在 backward 过程中立即清理 NaN/Inf 梯度
    # 这样 Adam 的二阶矩永远不会被污染
    _grad_clip_val = 1e4  # 逐参数梯度元素上限
    def _sanitize_grad(grad):
        if grad is None:
            return None
        g = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return g.clamp(-_grad_clip_val, _grad_clip_val)

    hooks = []
    for p in params.parameters():
        hooks.append(p.register_hook(_sanitize_grad))

    # 分组学习率：查找表 y 值用较低 lr
    table_params = []
    other_params = []
    for name, p in params.named_parameters():
        if '_y' in name:
            table_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': table_params, 'lr': lr_tables},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.1)

    traj_keys = [key for key, _gen in traj_list]

    losses = []
    training_history = []
    initial_params = {name: p.detach().clone() for name, p in params.named_parameters()}
    baseline_traj_losses = {}  # 第 1 epoch 记录各轨迹 baseline loss，用于归一化
    norm_floor = 1.0  # 归一化下限（第 1 epoch 结束后更新为中位数^alpha）
    norm_alpha = 0.5  # 软归一化指数：0=不归一化，1=完全归一化
    import time as _time
    t_start = _time.time()

    for epoch in range(n_epochs):
        t_epoch = _time.time()
        optimizer.zero_grad()
        epoch_loss = torch.tensor(0.0)
        traj_details = {}

        for traj_key, traj_gen in traj_list:
            traj = traj_gen()
            # 从轨迹本身提取速度（变速轨迹取初始速度）
            traj_speed = traj[0].v

            if sim_length is not None:
                max_t = sim_length / max(traj_speed, 0.1)
                traj = [p for p in traj if p.t <= max_t]
                if len(traj) < 10:
                    continue

            history = run_simulation(
                traj, init_speed=traj_speed,
                init_x=traj[0].x, init_y=traj[0].y, init_yaw=traj[0].theta,
                cfg=params.cfg,
                lat_ctrl=params.lat_ctrl, lon_ctrl=params.lon_ctrl,
                differentiable=True, tbptt_k=tbptt_k)

            loss, details = tracking_loss(history, ref_speed=traj_speed,
                                          w_lat=w_lat, w_head=w_head,
                                          w_speed=w_speed,
                                          w_steer_rate=w_steer_rate,
                                          w_acc_rate=w_acc_rate,
                                          return_details=True)

            # Per-trajectory loss 软归一化：baseline^alpha + 中位数下限
            if epoch == 0:
                baseline_traj_losses[traj_key] = max(loss.detach().item(), 1e-6)
            raw_baseline = baseline_traj_losses.get(traj_key, 1.0)
            norm_factor = max(raw_baseline ** norm_alpha, norm_floor)
            epoch_loss = epoch_loss + loss / norm_factor

            traj_details[traj_key] = details

        # 第 1 epoch 结束后，用中位数 baseline 设置归一化下限
        if epoch == 0 and len(baseline_traj_losses) > 1:
            sorted_baselines = sorted(baseline_traj_losses.values())
            median_baseline = sorted_baselines[len(sorted_baselines) // 2]
            norm_floor = median_baseline ** norm_alpha
            if verbose:
                print(f"  归一化下限: median_baseline={median_baseline:.4f}, "
                      f"norm_floor={norm_floor:.4f} (alpha={norm_alpha})")

        epoch_loss = epoch_loss / len(traj_list)

        # L2 正则化：惩罚参数偏离初始值，防止过拟合
        l2_reg = torch.tensor(0.0)
        for name, p in params.named_parameters():
            init_p = initial_params[name]
            l2_reg = l2_reg + ((p - init_p) ** 2).sum()
        epoch_loss = epoch_loss + 0.01 * l2_reg

        # 从 traj_details 计算各轨迹平均（兼容现有打印）
        epoch_details = {}
        if traj_details:
            all_keys = list(next(iter(traj_details.values())).keys())
            for k in all_keys:
                epoch_details[k] = sum(td[k] for td in traj_details.values()) / len(traj_details)

        epoch_loss.backward()

        # 梯度已由 hooks 清理 NaN/Inf，此处统计异常数量
        nan_count = 0
        for p in params.parameters():
            if p.grad is not None:
                nan_count += (p.grad.abs() >= _grad_clip_val).sum().item()

        # 全局梯度裁剪（防止爆炸梯度主导更新方向）
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params.parameters(), max_norm=grad_clip).item()
        optimizer.step()

        # 参数投影：物理约束
        with torch.no_grad():
            for name, p in params.named_parameters():
                if name in ('lon_ctrl.station_kp', 'lon_ctrl.station_ki',
                            'lon_ctrl.low_speed_kp', 'lon_ctrl.low_speed_ki',
                            'lon_ctrl.high_speed_kp', 'lon_ctrl.high_speed_ki'):
                    p.clamp_(min=0.0)
                elif name == 'lon_ctrl.switch_speed':
                    p.clamp_(min=0.5, max=10.0)
                elif name in ('lat_ctrl.T2_y', 'lat_ctrl.T3_y',
                              'lat_ctrl.T4_y', 'lat_ctrl.T6_y'):
                    # T2(预瞄时间)、T3(收敛时间)、T4(角速度预瞄)、T6(远预瞄时间)
                    # 均为时间相关参数，物理上不应为负
                    p.clamp_(min=0.0)

        scheduler.step()

        losses.append(epoch_loss.item())
        dt = _time.time() - t_epoch

        epoch_record = {
            'epoch': epoch + 1,
            'loss': epoch_loss.item(),
            'grad_norm': grad_norm,
            'nan_count': int(nan_count),
            'dt': dt,
            'per_trajectory': traj_details,
            'avg': dict(epoch_details),
        }
        training_history.append(epoch_record)

        if verbose:
            nan_warn = f" [!NaN grads:{int(nan_count)}]" if nan_count > 0 else ""
            print(f"[{epoch+1:3d}/{n_epochs}] loss={epoch_loss.item():8.4f} | "
                  f"lat_rmse={epoch_details['lat_rmse']:.4f}m "
                  f"head_rmse={epoch_details['head_rmse']:.4f}rad "
                  f"spd_rmse={epoch_details['speed_rmse']:.4f}m/s | "
                  f"grad_norm={grad_norm:.2f} "
                  f"dt={dt:.1f}s{nan_warn}",
                  flush=True)

        if verbose and len(traj_list) > 1:
            for traj_key in traj_keys:
                if traj_key in traj_details:
                    td = traj_details[traj_key]
                    print(f"    {traj_key:30s}: lat={td['lat_rmse']:.4f} head={td['head_rmse']:.4f} "
                          f"spd={td['speed_rmse']:.4f} | "
                          f"L_lat={td['loss_lat']:.3f} L_head={td['loss_head']:.3f} "
                          f"L_spd={td['loss_speed']:.3f}")

        if verbose and param_snapshot_interval > 0 and (epoch + 1) % param_snapshot_interval == 0:
            print(f"\n  --- 参数快照 (epoch {epoch+1}) ---")
            for name, p in params.named_parameters():
                init_val = initial_params[name]
                delta = p.detach() - init_val
                if p.numel() == 1:
                    pct = delta.item() / max(abs(init_val.item()), 1e-8) * 100
                    print(f"  {name:30s}: {init_val.item():.4f} -> {p.item():.4f} "
                          f"({delta.item():+.6f}, {pct:+.1f}%)")
                else:
                    print(f"  {name:30s}: max_delta={delta.abs().max().item():.6f} "
                          f"mean={p.detach().mean().item():.4f} "
                          f"[{p.detach().min().item():.3f}, {p.detach().max().item():.3f}]")
            print()

    # 清理梯度钩子
    for h in hooks:
        h.remove()

    total_time = _time.time() - t_start
    if verbose:
        print(f"\n训练完成! 总耗时: {total_time:.1f}s")
        print(f"  初始 loss: {losses[0]:.4f} → 最终 loss: {losses[-1]:.4f} "
              f"(Δ={losses[-1]-losses[0]:+.4f}, "
              f"{(losses[-1]-losses[0])/losses[0]*100:+.1f}%)")

    # 保存调参结果
    cfg_out = params.to_config_dict()
    type_names = trajectories if trajectories else TRAJECTORY_TYPES
    saved_path = save_tuned_config(cfg_out, meta={
        'final_loss': losses[-1],
        'initial_loss': losses[0],
        'epochs': n_epochs,
        'trajectory_types': type_names,
        'trajectory_count': len(traj_list),
        'speed_bands_kph': SPEED_BANDS_KPH,
        'lr': lr,
        'lr_tables': lr_tables,
        'tbptt_k': tbptt_k,
        'grad_clip': grad_clip,
        'w_lat': w_lat,
        'w_head': w_head,
        'w_speed': w_speed,
        'total_time_s': round(total_time, 1),
    })
    if verbose:
        print(f"参数已保存: {saved_path}")

    return {
        'losses': losses,
        'trajectory_types': trajectories if trajectories else TRAJECTORY_TYPES,
        'trajectory_keys': traj_keys,
        'training_history': training_history,
        'initial_params': {name: p.cpu().tolist() if p.numel() > 1 else p.item()
                           for name, p in initial_params.items()},
        'final_params': {name: p.detach().cpu().tolist() if p.numel() > 1 else p.detach().item()
                         for name, p in params.named_parameters()},
        'saved_path': saved_path,
        'params': params,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V2 可微调参训练')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-2,
                        help='主学习率（PID 增益等标量参数）')
    parser.add_argument('--lr-tables', type=float, default=5e-2,
                        help='查找表 y 值学习率')
    parser.add_argument('--trajectories', nargs='+', default=None,
                        help='轨迹类型名，自动展开到全速度段。'
                             '默认全量训练 (8×6=48 条)。可用类型: ' +
                             ', '.join(TRAJECTORY_TYPES))
    parser.add_argument('--sim-length', type=float, default=None,
                        help='仿真距离限制 (m)')
    parser.add_argument('--tbptt-k', type=int, default=150,
                        help='Truncated BPTT 窗口大小（步数），默认 150（3 秒）')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='全局梯度范数裁剪阈值')
    parser.add_argument('--snapshot-interval', type=int, default=10,
                        help='参数快照打印间隔（epoch 数）')
    parser.add_argument('--plant', type=str, default=None,
                        choices=['kinematic', 'dynamic', 'hybrid_dynamic', 'hybrid_v2'],
                        help='被控对象类型（覆盖 YAML 配置）')
    parser.add_argument('--config', type=str, default=None,
                        help='初始参数配置路径，用于 warm-start（从上次调参结果继续）')
    parser.add_argument('--w-lat', type=float, default=10.0,
                        help='横向误差 loss 权重')
    parser.add_argument('--w-head', type=float, default=8.0,
                        help='航向误差 loss 权重')
    parser.add_argument('--w-speed', type=float, default=3.0,
                        help='速度误差 loss 权重')
    parser.add_argument('--w-steer-rate', type=float, default=0.05,
                        help='转向平滑度惩罚权重')
    parser.add_argument('--w-acc-rate', type=float, default=0.01,
                        help='加速度平滑度惩罚权重')
    args = parser.parse_args()

    result = train(trajectories=args.trajectories, n_epochs=args.epochs,
                   lr=args.lr, lr_tables=args.lr_tables,
                   sim_length=args.sim_length,
                   tbptt_k=args.tbptt_k,
                   grad_clip=args.grad_clip,
                   param_snapshot_interval=args.snapshot_interval,
                   plant=args.plant,
                   config_path=args.config,
                   w_lat=args.w_lat, w_head=args.w_head,
                   w_speed=args.w_speed,
                   w_steer_rate=args.w_steer_rate,
                   w_acc_rate=args.w_acc_rate)
    print(f"\n最终 loss: {result['losses'][-1]:.6f}")
    print(f"保存路径: {result['saved_path']}")

    # 训练后自动化：loss 曲线、对比图、实验日志
    from optim.post_training import run_post_training
    hyperparams = {
        'epochs': args.epochs,
        'lr': args.lr,
        'lr_tables': args.lr_tables,
        'trajectory_types': result['trajectory_types'],
        'sim_length': args.sim_length,
        'tbptt_k': args.tbptt_k,
        'grad_clip': args.grad_clip,
        'plant': args.plant,
        'w_lat': args.w_lat,
        'w_head': args.w_head,
        'w_speed': args.w_speed,
    }
    run_post_training(result, hyperparams, plant=args.plant,
                      trajectory_types=args.trajectories)
