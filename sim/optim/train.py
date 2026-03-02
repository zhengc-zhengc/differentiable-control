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

from config import load_config, save_tuned_config
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from model.trajectory import (generate_straight, generate_circle,
                              generate_sine, generate_combined)
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
                  w_lat=10.0, w_head=5.0, w_speed=1.0,
                  w_steer_rate=0.01, w_acc_rate=0.01):
    """计算跟踪 loss：横向误差 + 航向误差 + 速度误差 + 平滑度惩罚。

    Args:
        history: run_simulation 返回的 differentiable=True 历史记录
        ref_speed: 参考速度 (m/s)
        w_lat/w_head/w_speed: 各误差项权重
        w_steer_rate/w_acc_rate: 平滑度惩罚权重

    Returns:
        loss (torch.Tensor): 标量 loss，支持 .backward()
    """
    lat_errs = torch.stack([h['lateral_error'] for h in history])
    head_errs = torch.stack([h['heading_error'] for h in history])
    speeds = torch.stack([h['v'] for h in history])
    steers = torch.stack([h['steer'] for h in history])
    accs = torch.stack([h['acc'] for h in history])

    speed_errs = speeds - ref_speed

    loss = (w_lat * (lat_errs ** 2).mean()
            + w_head * (head_errs ** 2).mean()
            + w_speed * (speed_errs ** 2).mean())

    if len(steers) > 1:
        steer_rate = steers[1:] - steers[:-1]
        acc_rate = accs[1:] - accs[:-1]
        loss = loss + w_steer_rate * (steer_rate ** 2).mean()
        loss = loss + w_acc_rate * (acc_rate ** 2).mean()

    return loss


# ---------- 轨迹生成器映射 ----------
_TRAJECTORY_BUILDERS = {
    'straight': lambda speed: generate_straight(length=200, speed=speed),
    'circle': lambda speed: generate_circle(radius=30.0, speed=speed,
                                            arc_angle=3.14159 / 2),
    'sine': lambda speed: generate_sine(amplitude=3.0, wavelength=50.0,
                                        n_waves=2, speed=speed),
    'combined': lambda speed: generate_combined(speed=speed),
}


def train(trajectories=None, n_epochs=100, lr=1e-3, lr_tables=5e-4,
          sim_length=None, sim_speed=5.0, verbose=True):
    """运行可微调参训练。

    Args:
        trajectories: 轨迹名列表，如 ['circle', 'sine', 'combined']
        n_epochs: 训练轮数
        lr: 主学习率（PID 增益等标量参数）
        lr_tables: 查找表 y 值学习率（通常低于主学习率）
        sim_length: 仿真距离限制 (m)，None 为全长
        sim_speed: 仿真速度 (m/s)
        verbose: 是否打印 epoch 信息

    Returns:
        dict: {'losses', 'saved_path', 'params'}
    """
    if trajectories is None:
        trajectories = ['circle', 'sine', 'combined']

    params = DiffControllerParams()

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

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        epoch_loss = torch.tensor(0.0)

        for traj_name in trajectories:
            builder = _TRAJECTORY_BUILDERS[traj_name]
            traj = builder(sim_speed)

            if sim_length is not None:
                max_t = sim_length / sim_speed
                traj = [p for p in traj if p.t <= max_t]
                if len(traj) < 10:
                    continue

            history = run_simulation(
                traj, init_speed=sim_speed, cfg=params.cfg,
                lat_ctrl=params.lat_ctrl, lon_ctrl=params.lon_ctrl,
                differentiable=True)

            loss = tracking_loss(history, ref_speed=sim_speed)
            epoch_loss = epoch_loss + loss

        epoch_loss = epoch_loss / len(trajectories)
        epoch_loss.backward()

        # NaN 梯度保护：长序列 BPTT 可能产生 NaN 梯度，替换为 0
        for p in params.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=10.0)
        optimizer.step()

        losses.append(epoch_loss.item())
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}  loss={epoch_loss.item():.6f}")

    # 保存调参结果
    cfg_out = params.to_config_dict()
    saved_path = save_tuned_config(cfg_out, meta={
        'final_loss': losses[-1],
        'epochs': n_epochs,
        'trajectories': trajectories,
        'lr': lr,
    })
    if verbose:
        print(f"参数已保存: {saved_path}")

    return {'losses': losses, 'saved_path': saved_path, 'params': params}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V2 可微调参训练')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='主学习率')
    parser.add_argument('--trajectories', nargs='+',
                        default=['circle', 'sine', 'combined'],
                        help='训练轨迹')
    parser.add_argument('--speed', type=float, default=5.0,
                        help='仿真速度 (m/s)')
    parser.add_argument('--sim-length', type=float, default=None,
                        help='仿真距离限制 (m)')
    args = parser.parse_args()

    result = train(trajectories=args.trajectories, n_epochs=args.epochs,
                   lr=args.lr, sim_speed=args.speed,
                   sim_length=args.sim_length)
    print(f"\n最终 loss: {result['losses'][-1]:.6f}")
    print(f"保存路径: {result['saved_path']}")
