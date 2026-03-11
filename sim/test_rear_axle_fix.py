# sim/test_rear_axle_fix.py
"""前后轴参考点修复效果对比。

通过 monkey-patch 切换 DynamicVehicle / HybridDynamicVehicle 的 x/y 属性，
分别用前轴坐标（修复前）和后轴坐标（修复后）跑完整闭环仿真，对比轨迹和横向误差。

用法:
    cd sim/
    python test_rear_axle_fix.py                        # 默认全部模型+全部场景
    python test_rear_axle_fix.py --plant dynamic        # 仅 dynamic
    python test_rear_axle_fix.py --plant hybrid_dynamic # 仅 hybrid_dynamic
"""
import argparse
import math
import os
import sys

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.trajectory import (generate_circle, generate_combined,
                               generate_lane_change,
                               generate_double_lane_change, generate_s_curve)
from model.dynamic_vehicle import DynamicVehicle
from model.hybrid_dynamic_vehicle import HybridDynamicVehicle
from sim_loop import run_simulation
from config import load_config


# ── monkey-patch 工具 ──────────────────────────────────────────────

def _front_axle_x(self):
    """返回前轴 x（修复前行为）。"""
    return self._state[0]

def _front_axle_y(self):
    """返回前轴 y（修复前行为）。"""
    return self._state[1]


def _patch_front_axle(cls):
    """模拟修复前行为：初始化不做后轴→前轴转换，x/y 直接返回 _state[0:2]。"""
    cls._orig_x = cls.x
    cls._orig_y = cls.y
    cls._orig_init = cls.__init__

    cls.x = property(_front_axle_x)
    cls.y = property(_front_axle_y)

    # Patch __init__: 撤销后轴→前轴坐标转换（模拟修复前的直接存储）
    original_init = cls._orig_init

    def _init_no_conversion(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # original_init 已做 rear→front 转换，这里反转回去
        L = self.dynamics.lf + self.dynamics.lr
        yaw = float(self._state[2])
        self._state = self._state.clone()
        self._state[0] -= L * math.cos(yaw)
        self._state[1] -= L * math.sin(yaw)

    cls.__init__ = _init_no_conversion


def _unpatch(cls):
    """恢复 cls.x / cls.y / __init__ 为修复后行为。"""
    cls.x = cls._orig_x
    cls.y = cls._orig_y
    cls.__init__ = cls._orig_init
    del cls._orig_x, cls._orig_y, cls._orig_init


def run_with_mode(traj, speed, cfg, mode):
    """用指定模式（front_axle / rear_axle）跑仿真。"""
    model_type = cfg['vehicle'].get('model_type', 'kinematic')
    cls = HybridDynamicVehicle if model_type == 'hybrid_dynamic' else DynamicVehicle

    if mode == 'front_axle':
        _patch_front_axle(cls)

    try:
        history = run_simulation(traj, init_speed=speed, cfg=cfg)
    finally:
        if mode == 'front_axle':
            _unpatch(cls)

    return history


def to_float(v):
    if isinstance(v, torch.Tensor):
        return v.item()
    return float(v)


# ── 场景定义 ──────────────────────────────────────────────────────

SCENARIOS = [
    ('圆弧 R=15m, 5kph',
     lambda: generate_circle(radius=15.0, speed=5.0 / 3.6, arc_angle=math.pi),
     5.0 / 3.6),
    ('圆弧 R=30m, 5m/s',
     lambda: generate_circle(radius=30.0, speed=5.0, arc_angle=math.pi),
     5.0),
    ('圆弧 R=50m, 5m/s',
     lambda: generate_circle(radius=50.0, speed=5.0, arc_angle=math.pi / 2),
     5.0),
    ('组合 5m/s',
     lambda: generate_combined(speed=5.0),
     5.0),
    ('换道 5m/s',
     lambda: generate_lane_change(lane_width=3.5, change_length=50.0, speed=5.0),
     5.0),
    ('双换道 5m/s',
     lambda: generate_double_lane_change(lane_width=3.5, change_length=50.0, speed=5.0),
     5.0),
    ('S弯 R=50m, 5m/s',
     lambda: generate_s_curve(radius=50.0, arc_angle=math.pi / 4, speed=5.0),
     5.0),
]


def plot_comparison(plant_type, output_dir):
    """对一种被控对象，画修复前/后对比图。"""
    cfg = load_config()
    cfg['vehicle']['model_type'] = plant_type

    n = len(SCENARIOS)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    fig.suptitle(
        f'前轴输出（修复前）vs 后轴输出（修复后）—— {plant_type} 模型\n'
        f'（两次独立闭环仿真，控制器看到不同坐标 → 不同控制决策）',
        fontsize=14, fontweight='bold')

    print(f'\n{"=" * 60}')
    print(f'  {plant_type} 模型：修复前后对比')
    print(f'{"=" * 60}')
    header = f'  {"场景":<22s} {"前轴 RMSE":>12s} {"后轴 RMSE":>12s} {"改善":>8s}'
    print(header)
    print(f'  {"-" * 56}')

    for i, (name, traj_fn, speed) in enumerate(SCENARIOS):
        traj = traj_fn()

        # 修复前：前轴坐标
        h_before = run_with_mode(traj, speed, cfg, 'front_axle')
        # 修复后：后轴坐标
        h_after = run_with_mode(traj, speed, cfg, 'rear_axle')

        t_b = [h['t'] for h in h_before]
        t_a = [h['t'] for h in h_after]
        lat_b = [to_float(h['lateral_error']) for h in h_before]
        lat_a = [to_float(h['lateral_error']) for h in h_after]
        x_b = [to_float(h['x']) for h in h_before]
        y_b = [to_float(h['y']) for h in h_before]
        x_a = [to_float(h['x']) for h in h_after]
        y_a = [to_float(h['y']) for h in h_after]

        rmse_b = (sum(e ** 2 for e in lat_b) / len(lat_b)) ** 0.5
        rmse_a = (sum(e ** 2 for e in lat_a) / len(lat_a)) ** 0.5
        pct = (rmse_a - rmse_b) / rmse_b * 100 if rmse_b > 1e-8 else 0.0

        print(f'  {name:<22s} {rmse_b:>12.4f} {rmse_a:>12.4f} {pct:>+7.1f}%')

        # 左图：轨迹
        ax_traj = axes[i, 0]
        ref_x = [p.x for p in traj]
        ref_y = [p.y for p in traj]
        ax_traj.plot(ref_x, ref_y, 'k--', label='参考轨迹', linewidth=1.5, alpha=0.7)
        ax_traj.plot(x_b, y_b, 'b-', label=f'修复前 (RMSE={rmse_b:.3f}m)',
                     linewidth=1.2, alpha=0.7)
        ax_traj.plot(x_a, y_a, 'r-', label=f'修复后 (RMSE={rmse_a:.3f}m)',
                     linewidth=1.2, alpha=0.7)
        ax_traj.set_title(f'{name} — 轨迹', fontsize=11)
        ax_traj.set_xlabel('x (m)')
        ax_traj.set_ylabel('y (m)')
        ax_traj.set_aspect('equal', adjustable='datalim')
        ax_traj.legend(fontsize=9)
        ax_traj.grid(True, alpha=0.3)

        # 右图：横向误差
        ax_lat = axes[i, 1]
        ax_lat.plot(t_b, lat_b, 'b-',
                    label=f'修复前 RMSE={rmse_b:.3f}m', alpha=0.8)
        ax_lat.plot(t_a, lat_a, 'r-',
                    label=f'修复后 RMSE={rmse_a:.3f}m', alpha=0.8)
        ax_lat.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax_lat.set_title(f'{name} — 横向误差', fontsize=11)
        ax_lat.set_xlabel('时间 (s)')
        ax_lat.set_ylabel('横向误差 (m)')
        ax_lat.legend(fontsize=9)
        ax_lat.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'rear_axle_fix_{plant_type}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'\n  图片已保存: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description='前后轴修复对比测试')
    parser.add_argument('--plant', choices=['dynamic', 'hybrid_dynamic'],
                        default=None, help='指定模型（默认两种都跑）')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results', 'baseline')

    plants = [args.plant] if args.plant else ['dynamic', 'hybrid_dynamic']
    for p in plants:
        plot_comparison(p, os.path.join(output_dir, p))


if __name__ == '__main__':
    main()
