"""对所有 49 个标准评估场景画目标 MLP 的输出可视化。

并行：复用 run_simulation_batch(hard_mode=True, capture_mlp=True) 一次性把
49 条轨迹推进完，把每步 MLP 输入/输出按 batch 维度拆开，再串行画 panel。
每场景一张 3x3 panel，无变体对比——只画目标 MLP。

布局（4x3）：
  行 1：轨迹（参考 vs 实际） | 横向跟踪误差 | 输入 OOD 距离
  行 2：MLP 牵引车速度残差 vx_t / vy_t / r_t
  行 3：MLP 挂车速度残差 vx_s / vy_s / r_s
  行 4：MLP 相对位姿残差 rel_x / rel_y / rel_yaw

用法：
  python sim/debug/plot_mlp_outputs_all_scenarios.py \
      --ckpt configs/checkpoints/best_truck_trailer_error_model_train_loss_0518.pth \
      --subdir 0518 --label 0518TL
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, SIM_DIR)

from config import apply_plant_override, load_config
from model.trajectory import expand_trajectories, generate_park_route
from optim.train_batch import run_simulation_batch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# v2 MLP 9D 输出语义
OUTPUT_NAMES = ['vx_t (m/s)', 'vy_t (m/s)', 'r_t (rad/s)',
                'vx_s (m/s)', 'vy_s (m/s)', 'r_s (rad/s)',
                'rel_x (m)', 'rel_y (m)', 'rel_yaw (rad)']


def build_scenarios(trajectory_types: list[str] | None):
    """[(key, label, generator), ...]，末尾追加 park_route。"""
    expanded = expand_trajectories(trajectory_types)
    scenarios = [(k, lbl, gen) for k, lbl, gen in expanded]
    if trajectory_types is None or 'park_route' in (trajectory_types or []):
        scenarios.append(('park_route', '园区综合', generate_park_route))
    return scenarios


def plot_one_scenario(scenario_key: str, scenario_label: str,
                      arr: dict, label_tag: str, out_dir: str):
    """画单场景 3x3 panel。

    arr 包含：
        t, x, y, ref_x, ref_y, lat_err, head_err, steer
        mlp_input_raw [T, D_in], mlp_output_clipped [T, D_out]
        feature_mean, feature_scale, motion_error_clip
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'{label_tag} MLP 输出 — {scenario_label}',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.30)

    t = arr['t']
    out_clip = arr['mlp_output_clipped']  # [T, D_out]

    # (1,1) 轨迹
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(arr['ref_x'], arr['ref_y'], 'k--', linewidth=1.0,
            alpha=0.6, label='参考轨迹')
    ax.plot(arr['x'], arr['y'], '-', linewidth=1.5,
            color='#e41a1c', alpha=0.9, label='实际轨迹')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('轨迹（参考 vs 实际）')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.3)

    # (1,2) 横向跟踪误差
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, arr['lat_err'], '-', linewidth=1.2,
            color='#e41a1c', alpha=0.9)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    lat_rmse = float(np.sqrt(np.mean(arr['lat_err'] ** 2)))
    lat_max = float(np.max(np.abs(arr['lat_err'])))
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('横向误差 (m)')
    ax.set_title(f'横向跟踪误差  (RMSE={lat_rmse:.3f}, max={lat_max:.3f})')
    ax.grid(True, alpha=0.3)

    # (1,3) 输入 OOD 距离（z-score）
    ax = fig.add_subplot(gs[0, 2])
    if arr.get('feature_mean') is not None:
        inp = arr['mlp_input_raw']  # [T, D]
        z = (inp - arr['feature_mean']) / arr['feature_scale']
        z_max = np.max(np.abs(z), axis=1)
        z_l2 = np.sqrt(np.mean(z * z, axis=1))
        t_mlp = np.arange(z.shape[0]) * arr['dt']
        ax.plot(t_mlp, z_max, '-', linewidth=0.9, color='#e41a1c',
                label='|z|_∞（最大单维偏离）')
        ax.plot(t_mlp, z_l2, '-', linewidth=0.9, color='#377eb8',
                label='|z|_2 / √D（均方）')
        ax.axhline(2, color='gray', linewidth=0.5, linestyle=':',
                   label='z=2 阈值')
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('归一化距离')
        ax.set_title('输入 OOD 距离（vs 训练分布）')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, '无归一化统计（checkpoint 缺 feature_mean）',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('输入 OOD 距离')

    # 行 2/3/4：完整 9D MLP 输出
    t_mlp = np.arange(out_clip.shape[0]) * arr['dt']
    clip_vec = arr.get('motion_error_clip')  # [D_out] or None
    n_out = out_clip.shape[1]
    for out_idx in range(n_out):
        row = 1 + out_idx // 3
        col = out_idx % 3
        ax = fig.add_subplot(gs[row, col])
        col_data = out_clip[:, out_idx]
        ax.plot(t_mlp, col_data, '-', linewidth=0.9,
                color='#e41a1c', alpha=0.9, label=f'{label_tag} MLP 输出')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        if clip_vec is not None:
            cv = float(clip_vec[out_idx])
            ax.axhline(cv, color='#ff7f00', linewidth=0.6, linestyle=':',
                       alpha=0.8, label=f'clip ±{cv:.4g}')
            ax.axhline(-cv, color='#ff7f00', linewidth=0.6, linestyle=':',
                       alpha=0.8)
        out_rms = float(np.sqrt(np.mean(col_data ** 2)))
        out_absmax = float(np.max(np.abs(col_data))) if col_data.size else 0.0
        ax.set_xlabel('时间 (s)'); ax.set_ylabel(f'残差 {OUTPUT_NAMES[out_idx]}')
        ax.set_title(f'MLP 输出：{OUTPUT_NAMES[out_idx]}  '
                     f'(RMS={out_rms:.4g}, |·|max={out_absmax:.4g})')
        ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, f'panel_{scenario_key}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', required=True,
                        help='MLP checkpoint 路径（相对 sim/ 或绝对）')
    parser.add_argument('--config', default=None,
                        help='基线 YAML，None 走 default.yaml；'
                             '建议传匹配 ckpt 的 train_with_*.yaml')
    parser.add_argument('--subdir', default=None,
                        help='输出子目录名；None 时从 ckpt 文件名末尾数字推断')
    parser.add_argument('--label', default=None,
                        help='图中 MLP 标签（如 0518TL）；None 时同 subdir')
    parser.add_argument('--out-dir-root', default=None,
                        help='输出根目录；默认 sim/results/diagnostic/mlp_output_panels/')
    parser.add_argument('--trajectories', nargs='+', default=None,
                        help='可选筛选标准类型名；None 跑全量 8×6 + park_route = 49')
    args = parser.parse_args()

    # 推断 subdir / label
    ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]
    if args.subdir is None:
        import re as _re
        m = _re.search(r'_(\d{3,})$', ckpt_base)
        args.subdir = m.group(1) if m else ckpt_base
    if args.label is None:
        args.label = args.subdir

    out_root = args.out_dir_root or os.path.join(
        SIM_DIR, 'results', 'diagnostic', 'mlp_output_panels')
    out_dir = os.path.join(out_root, args.subdir)
    os.makedirs(out_dir, exist_ok=True)

    # 配置：用 train_with_<subdir>.yaml 兜底 default.yaml，确保
    # truck_trailer_vehicle 段（rear_drive_torque 等）和 ckpt 训练时一致
    cfg = load_config(args.config) if args.config else load_config()
    apply_plant_override(cfg, 'truck_trailer')
    cfg['truck_trailer_vehicle']['checkpoint_path'] = args.ckpt

    print(f'被测 MLP：{args.ckpt}')
    print(f'输出目录：{out_dir}')
    print(f'图标签：{args.label}')

    scenarios = build_scenarios(args.trajectories)
    print(f'场景数：{len(scenarios)}')

    # 一次并行跑完所有场景
    print('\n--- 并行仿真（hard_mode + capture_mlp）---')
    trajs = [gen() for _k, _lbl, gen in scenarios]
    init_vs = [float(t[0].v) for t in trajs]
    t0 = time.time()
    hist = run_simulation_batch(trajs, cfg=cfg, tbptt_k=0,
                                hard_mode=True, capture_mlp=True)
    print(f'  仿真完成：{time.time() - t0:.1f}s ({len(scenarios)} 场景）')

    mlp_hist = hist.get('mlp_history')
    if mlp_hist is None:
        raise RuntimeError('mlp_history 为空——确认 ckpt 是否成功加载')
    print(f"  MLP 输入维度 {mlp_hist['input_dim']}, 输出维度 {mlp_hist['output_dim']}")

    # 按 batch 维度拆开，串行画图
    print('\n--- 串行绘图 ---')
    dt = cfg['simulation']['dt']
    valid_mask = hist['valid_mask']  # [B, T_max]

    feature_mean = (mlp_hist['feature_mean'].squeeze(0).cpu().numpy()
                    if mlp_hist['feature_mean'] is not None else None)
    feature_scale = (mlp_hist['feature_scale'].squeeze(0).cpu().numpy()
                     if mlp_hist['feature_scale'] is not None else None)
    motion_error_clip = (mlp_hist['motion_error_clip'].squeeze(0).cpu().numpy()
                         if mlp_hist['motion_error_clip'] is not None else None)

    for idx, ((key, label, _gen), traj) in enumerate(zip(scenarios, trajs)):
        n_valid = int(valid_mask[idx].sum().item())
        arr = {
            'dt': dt,
            't': np.arange(n_valid) * dt,
            'x': hist['x'][idx, :n_valid].cpu().numpy(),
            'y': hist['y'][idx, :n_valid].cpu().numpy(),
            'ref_x': hist['ref_x'][idx, :n_valid].cpu().numpy(),
            'ref_y': hist['ref_y'][idx, :n_valid].cpu().numpy(),
            'lat_err': hist['lateral_error'][idx, :n_valid].cpu().numpy(),
            'head_err': hist['heading_error'][idx, :n_valid].cpu().numpy(),
            'steer': hist['steer'][idx, :n_valid].cpu().numpy(),
            'mlp_input_raw': mlp_hist['input_raw'][idx, :n_valid].cpu().numpy(),
            'mlp_output_clipped': mlp_hist['output_clipped'][idx, :n_valid].cpu().numpy(),
            'feature_mean': feature_mean,
            'feature_scale': feature_scale,
            'motion_error_clip': motion_error_clip,
        }
        path = plot_one_scenario(key, label, arr, args.label, out_dir)
        print(f'  [{idx+1}/{len(scenarios)}] {label} → {os.path.basename(path)}')

    print(f'\n全部 panel 已存：{out_dir}')


if __name__ == '__main__':
    main()
