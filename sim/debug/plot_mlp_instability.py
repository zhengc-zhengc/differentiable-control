"""可视化 0507 MLP 失控原因。

读取 investigate_mlp_instability.py 落下的 npz 数据，画出：
  1. 轨迹对比 + 何时开始偏离 baseline
  2. MLP 输出（牵引车速度残差三分量）时序
  3. 组件消融柱状图（每个变体的横向 RMSE）
  4. 首步 MLP 输出量级（证明还没建立误差就已经"乱说话"）
  5. 输入特征 OOD 距离 vs MLP 输出量级 vs 时间
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 0507 MLP 的特征归一化统计
FEATURE_MEAN = np.array([0.0, 0.0, 5.115, 0.006, 0.006, 5.115, 0.006, 0.006,
                         0.0, 0.0, 0.0, 1.0, 0.294, 1485.873])
FEATURE_SCALE = np.array([1.0, 1.0, 2.9812, 0.2363, 0.0292, 2.9812, 0.2363,
                          0.0292, 1.0, 1.0, 1.0, 1.0, 1.0082, 1933.2778])
FEATURE_NAMES = ['trailer_mass', 'has_trailer', 'vx_t', 'vy_t', 'r_t',
                 'vx_s', 'vy_s', 'r_s', 'rel_x', 'rel_y',
                 'sin_rel_yaw', 'cos_rel_yaw', 'steer_sw', 'rear_torque']

OUTPUT_NAMES = ['vx_t (m/s)', 'vy_t (m/s)', 'r_t (rad/s)',
                'vx_s', 'vy_s', 'r_s',
                'rel_x', 'rel_y', 'rel_yaw']

VARIANTS_FULL = ['no_mlp', 'mlp_default', 'mlp_0507_full',
                 'mlp_0507_zero_vel_t', 'mlp_0507_zero_pose_s',
                 'mlp_0507_only_yaw_t', 'mlp_0507_only_vx_t',
                 'mlp_0507_only_vy_t']
VARIANT_LABELS = {
    'no_mlp': '无 MLP（纯 RK4）',
    'mlp_default': '默认 MLP (64 隐层)',
    'mlp_0507_full': '0507 MLP 完整',
    'mlp_0507_zero_vel_t': '0507 但牵引车 v 残差置零',
    'mlp_0507_zero_pose_s': '0507 但相对位姿残差置零',
    'mlp_0507_only_yaw_t': '0507 仅留 r_t',
    'mlp_0507_only_vx_t': '0507 仅留 vx_t',
    'mlp_0507_only_vy_t': '0507 仅留 vy_t',
}
COLORS = {
    'no_mlp': '#000000',
    'mlp_default': '#377eb8',
    'mlp_0507_full': '#e41a1c',
    'mlp_0507_zero_vel_t': '#4daf4a',
    'mlp_0507_zero_pose_s': '#984ea3',
    'mlp_0507_only_yaw_t': '#ff7f00',
    'mlp_0507_only_vx_t': '#a65628',
    'mlp_0507_only_vy_t': '#f781bf',
}


def load_scenario(base, scenario_key):
    sd = os.path.join(base, scenario_key)
    data = {}
    for v in VARIANTS_FULL:
        path = os.path.join(sd, f'{v}.npz')
        if not os.path.exists(path):
            continue
        d = dict(np.load(path, allow_pickle=True))
        data[v] = d
    return data


# ---- 图 1：每场景一张（轨迹 + 偏离时序 + MLP vy_t 残差） ----
def plot_scenario_panel(scenario_key, scenario_label, data, out_dir):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f'0507 MLP 失控诊断 — {scenario_label}',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.30)

    no = data['no_mlp']
    full = data['mlp_0507_full']
    zero_vel = data['mlp_0507_zero_vel_t']

    # (1,1) 轨迹对比
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(no['ref_traj_x'], no['ref_traj_y'], 'k--',
            linewidth=1.0, alpha=0.6, label='参考轨迹')
    for v in ['no_mlp', 'mlp_default', 'mlp_0507_full',
              'mlp_0507_zero_vel_t']:
        if v not in data:
            continue
        d = data[v]
        ax.plot(d['hist_x'], d['hist_y'], '-', linewidth=1.4,
                color=COLORS[v], alpha=0.85, label=VARIANT_LABELS[v])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('轨迹对比')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)

    # (1,2) 偏离 no_mlp 基线的 (x,y) 距离
    ax = fig.add_subplot(gs[0, 1])
    n = min(len(no['hist_t']), len(full['hist_t']))
    t = no['hist_t'][:n]
    for v in ['mlp_default', 'mlp_0507_full', 'mlp_0507_zero_vel_t']:
        if v not in data:
            continue
        d = data[v]
        m = min(n, len(d['hist_t']))
        dx = d['hist_x'][:m] - no['hist_x'][:m]
        dy = d['hist_y'][:m] - no['hist_y'][:m]
        dist = np.sqrt(dx*dx + dy*dy)
        ax.plot(t[:m], dist, '-', linewidth=1.3,
                color=COLORS[v], alpha=0.85, label=VARIANT_LABELS[v])
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('与无 MLP 基线偏差 (m)')
    ax.set_title('轨迹偏离速度（log 轴）')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    # (1,3) 横向跟踪误差时序
    ax = fig.add_subplot(gs[0, 2])
    for v in ['no_mlp', 'mlp_default', 'mlp_0507_full',
              'mlp_0507_zero_vel_t']:
        if v not in data:
            continue
        d = data[v]
        ax.plot(d['hist_t'], d['hist_lat_err'], '-', linewidth=1.3,
                color=COLORS[v], alpha=0.85, label=VARIANT_LABELS[v])
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('横向误差 (m)')
    ax.set_title('横向跟踪误差')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (2,*) MLP 牵引车速度残差三分量时序
    out_full = full['mlp_output_clipped']
    n_steps = out_full.shape[0]
    t_mlp = np.arange(n_steps) * 0.02

    for i, comp in enumerate([0, 1, 2]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t_mlp, out_full[:, comp], '-', linewidth=0.8,
                color=COLORS['mlp_0507_full'], alpha=0.85, label='0507')
        if 'mlp_default' in data:
            d = data['mlp_default']
            out_def = d['mlp_output_clipped']
            t_def = np.arange(out_def.shape[0]) * 0.02
            ax.plot(t_def, out_def[:, comp], '-', linewidth=0.8,
                    color=COLORS['mlp_default'], alpha=0.6,
                    label='默认 MLP')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        # 训练标尺：clip 范围
        if comp == 0:
            ax.axhline(0.148, color='#ff7f00', linewidth=0.6, linestyle=':',
                       alpha=0.8, label='0507 clip ±')
            ax.axhline(-0.148, color='#ff7f00', linewidth=0.6, linestyle=':',
                       alpha=0.8)
        elif comp == 1:
            ax.axhline(0.539, color='#ff7f00', linewidth=0.6, linestyle=':',
                       alpha=0.8, label='0507 clip ±')
            ax.axhline(-0.539, color='#ff7f00', linewidth=0.6, linestyle=':',
                       alpha=0.8)
        elif comp == 2:
            ax.axhline(0.00762, color='#ff7f00', linewidth=0.6, linestyle=':',
                       alpha=0.8, label='0507 clip ±')
            ax.axhline(-0.00762, color='#ff7f00', linewidth=0.6, linestyle=':',
                       alpha=0.8)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel(f'残差 {OUTPUT_NAMES[comp]}')
        ax.set_title(f'MLP 输出：牵引车 {OUTPUT_NAMES[comp]}')
        ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)

    # (3,1) 输入 OOD 距离（z-score L∞ norm）随时间
    ax = fig.add_subplot(gs[2, 0])
    inp_full = full['mlp_input_raw']
    z = (inp_full - FEATURE_MEAN) / FEATURE_SCALE
    z_max = np.max(np.abs(z), axis=1)
    z_l2 = np.sqrt(np.mean(z * z, axis=1))
    ax.plot(t_mlp, z_max, '-', linewidth=0.9, color='#e41a1c',
            label='|z|_∞（最大单维偏离）')
    ax.plot(t_mlp, z_l2, '-', linewidth=0.9, color='#377eb8',
            label='|z|_2 / √14（均方）')
    ax.axhline(2, color='gray', linewidth=0.5, linestyle=':',
               label='z=2 阈值')
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('归一化距离')
    ax.set_title('输入 OOD 距离（vs 训练分布）')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (3,2) 控制器命令对比
    ax = fig.add_subplot(gs[2, 1])
    for v in ['no_mlp', 'mlp_0507_full', 'mlp_0507_zero_vel_t']:
        if v not in data:
            continue
        d = data[v]
        ax.plot(d['hist_t'], d['hist_steer'], '-', linewidth=1.0,
                color=COLORS[v], alpha=0.85, label=VARIANT_LABELS[v])
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('转角命令 (deg, 方向盘)')
    ax.set_title('转向命令时序')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (3,3) 组件消融 RMSE 柱状图
    ax = fig.add_subplot(gs[2, 2])
    bars = []
    for v in VARIANTS_FULL:
        if v not in data:
            bars.append(np.nan); continue
        bars.append(np.sqrt(np.mean(data[v]['hist_lat_err'] ** 2)))
    x = np.arange(len(bars))
    colors_b = [COLORS[v] for v in VARIANTS_FULL]
    ax.bar(x, bars, color=colors_b, edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    short_labels = ['无MLP', '默认', '0507\n完整',
                    '0507\n零Vt', '0507\n零RelPose', '0507\n仅r_t',
                    '0507\n仅vx_t', '0507\n仅vy_t']
    ax.set_xticklabels(short_labels, fontsize=8, rotation=0)
    ax.set_ylabel('横向 RMSE (m)')
    ax.set_title('组件消融：横向 RMSE')
    ax.grid(True, alpha=0.3, axis='y')

    out_path = os.path.join(out_dir, f'panel_{scenario_key}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")
    return out_path


# ---- 图 2：早期失控时序（前 100 步：MLP 输出 vs 累积偏差） ----
def plot_early_steps(scenario_key, scenario_label, data, out_dir,
                     n_steps=200):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'早期失控时序（前 {n_steps} 步） — {scenario_label}',
                 fontsize=14, fontweight='bold')

    full = data['mlp_0507_full']
    no = data['no_mlp']

    n = min(n_steps, full['mlp_output_clipped'].shape[0],
            len(full['hist_t']), len(no['hist_t']))
    t = np.arange(n) * 0.02
    out_full = full['mlp_output_clipped']

    # (0,0) MLP 输出三分量（前 N 步）
    ax = axes[0, 0]
    for comp, name, color in [(0, 'vx_t', '#377eb8'),
                              (1, 'vy_t', '#e41a1c'),
                              (2, 'r_t', '#4daf4a')]:
        ax.plot(t, out_full[:n, comp], '-', linewidth=1.2,
                color=color, label=name, alpha=0.9)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('MLP 残差')
    ax.set_title('MLP 牵引车速度三分量（早期 N 步）')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # (0,1) 累积位置偏差
    ax = axes[0, 1]
    dx = full['hist_x'][:n] - no['hist_x'][:n]
    dy = full['hist_y'][:n] - no['hist_y'][:n]
    dist = np.sqrt(dx*dx + dy*dy)
    ax.plot(t, dist, '-', linewidth=1.3, color='#e41a1c',
            label='|0507 - no_mlp|')
    ax.plot(t, np.abs(dx), '--', linewidth=0.9, color='#377eb8',
            label='|Δx|', alpha=0.8)
    ax.plot(t, np.abs(dy), '--', linewidth=0.9, color='#4daf4a',
            label='|Δy|', alpha=0.8)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('累积偏差 (m)')
    ax.set_title('与无 MLP 基线的位置偏差（早期）')
    ax.set_yscale('log')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which='both')

    # (1,0) 状态对比：vy_t（base 出来 vs 实际状态）
    ax = axes[1, 0]
    state_after = full['state_after_mlp'][:n]
    state_before = full['state_before_mlp'][:n]
    no_after = no['state_after_mlp'][:n]
    ax.plot(t, state_after[:, 4], '-', linewidth=1.2, color='#e41a1c',
            label='0507 实际 vy_t', alpha=0.9)
    ax.plot(t, state_before[:, 4], '-', linewidth=0.8, color='#ff7f00',
            label='0507 RK4 base 输出 vy_t', alpha=0.7)
    ax.plot(t, no_after[:, 4], '-', linewidth=1.2, color='#000000',
            label='no_mlp vy_t', alpha=0.9)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.3)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('vy_t (m/s)')
    ax.set_title('车体侧向速度 vy_t：0507 注入 vs 真实物理')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # (1,1) 控制器立刻反应
    ax = axes[1, 1]
    ax.plot(t, full['hist_steer'][:n], '-', linewidth=1.0,
            color='#e41a1c', label='0507 转角')
    ax.plot(t, no['hist_steer'][:n], '-', linewidth=1.0,
            color='#000000', label='no_mlp 转角', alpha=0.7)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('转角命令 (deg)')
    ax.set_title('转向命令早期对比（feedback loop 起步）')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'early_{scenario_key}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---- 图 3：跨场景 ablation 总览 ----
def plot_cross_scenario_summary(all_data, out_dir):
    scenarios = list(all_data.keys())
    n_scen = len(scenarios)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('0507 MLP 失控成因 — 组件消融跨场景总览',
                 fontsize=15, fontweight='bold')

    # (0) RMSE 柱状图：每个场景下每个变体的横向 RMSE
    ax = axes[0]
    n_var = len(VARIANTS_FULL)
    width = 0.10
    x = np.arange(n_scen)
    offsets = np.linspace(-width * (n_var - 1) / 2,
                          width * (n_var - 1) / 2, n_var)
    for vi, v in enumerate(VARIANTS_FULL):
        rmses = []
        for s in scenarios:
            d = all_data[s].get(v)
            if d is None:
                rmses.append(0); continue
            rmses.append(np.sqrt(np.mean(d['hist_lat_err'] ** 2)))
        ax.bar(x + offsets[vi], rmses, width, color=COLORS[v],
               edgecolor='black', linewidth=0.4,
               label=VARIANT_LABELS[v])
    ax.set_xticks(x)
    scen_labels = {
        'straight_5kph': '直行 5kph',
        'circle_25kph_R80': '圆周 25kph (R=80)',
        'lane_change_5kph': '变道 5kph',
        'clothoid_left_5kph': 'clothoid 左转 5kph',
    }
    ax.set_xticklabels([scen_labels.get(s, s) for s in scenarios],
                       fontsize=10)
    ax.set_ylabel('横向 RMSE (m)（log 轴）')
    ax.set_yscale('log')
    ax.set_title('横向跟踪 RMSE：每场景 8 变体对比')
    ax.legend(fontsize=8, ncol=4, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', which='both')

    # (1) MLP 牵引车速度残差时序的 RMS 量级（仅 0507 完整）
    ax = axes[1]
    ax.set_title('0507 MLP 输出量级：牵引车速度残差三分量 RMS（每场景）')
    components = [(0, 'vx_t (m/s)'), (1, 'vy_t (m/s)'),
                  (2, 'r_t (rad/s)')]
    width = 0.25
    offsets = np.linspace(-width, width, 3)
    for ci, (comp_idx, comp_name) in enumerate(components):
        rmses = []
        for s in scenarios:
            d = all_data[s].get('mlp_0507_full')
            if d is None:
                rmses.append(0); continue
            out = d['mlp_output_clipped']
            rmses.append(np.sqrt(np.mean(out[:, comp_idx] ** 2)))
        color = ['#377eb8', '#e41a1c', '#4daf4a'][ci]
        ax.bar(x + offsets[ci], rmses, width, color=color,
               edgecolor='black', linewidth=0.4, label=comp_name)
    ax.set_xticks(x)
    ax.set_xticklabels([scen_labels.get(s, s) for s in scenarios],
                       fontsize=10)
    ax.set_ylabel('RMS 输出量级')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', which='both')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'cross_scenario_summary.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---- 图 4：MLP 静态测试（喂任意输入，看输出）====
def plot_mlp_static_test(out_dir):
    """直接前向 0507 MLP，扫描输入空间。验证：
    1. 喂训练分布均值，输出应≈0
    2. 喂5kph直行输入，输出有多偏离 0
    3. 扫描 vx_t 从 0 到 15 m/s，看 9D 输出曲线
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import model.truck_trailer_vehicle as ttv
    from config import load_config, apply_plant_override
    from model.vehicle_factory import create_vehicle

    cfg = load_config('configs/train_with_0507.yaml')
    apply_plant_override(cfg, 'truck_trailer')
    car = create_vehicle(cfg, x=0, y=0, yaw=0, v=0, dt=0.02,
                         differentiable=False)
    mlp = car._mlp
    f_mean = car._feature_mean
    f_scale = car._feature_scale
    clip = car._motion_error_clip

    print('  MLP 输入维度', car._mlp_input_dim, '输出维度', car._mlp_output_dim)
    print('  feature_mean:', f_mean.flatten().tolist())
    print('  feature_scale:', f_scale.flatten().tolist())
    print('  motion_error_clip:', clip.flatten().tolist())

    # ---- 测试 1：喂训练均值，看输出
    inp_mean = torch.zeros(1, 14)  # 归一化后的 0 = 训练均值
    with torch.no_grad():
        out_mean = mlp(inp_mean)
    print('\n  喂归一化输入=0（训练均值），MLP 输出：')
    for i, n in enumerate(OUTPUT_NAMES):
        print(f'    {n}: {out_mean[0,i].item():+.6f}')

    # ---- 测试 2：扫描 vx_t（无挂车，无横向运动，无控制）
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle('0507 MLP 静态扫描：喂"无挂车 + 无横向运动 + 无控制"输入',
                 fontsize=14, fontweight='bold')
    vx_grid = np.linspace(0.0, 15.0, 200)  # 0 - 54 kph

    raw_inputs = []
    for vx in vx_grid:
        # 14D：[trailer_mass=0, has_trailer=0, vx_t, vy_t=0, r_t=0,
        #       vx_s, vy_s=0, r_s=0, rel_x=0, rel_y=0,
        #       sin_rel=0, cos_rel=1, steer_sw=0, rear_torque=0]
        raw = np.array([0.0, 0.0, vx, 0.0, 0.0, vx, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        raw_inputs.append(raw)
    raw_inputs = np.array(raw_inputs)
    norm_inputs = (raw_inputs - FEATURE_MEAN) / FEATURE_SCALE
    with torch.no_grad():
        out = mlp(torch.tensor(norm_inputs, dtype=torch.float32)).numpy()

    # 9 个 subplot：每分量随 vx 的曲线
    for i in range(9):
        r, c = i // 3, i % 3
        ax = axes[r, c]
        clip_val = clip.flatten()[i].item()
        ax.plot(vx_grid * 3.6, out[:, i], '-', linewidth=1.5,
                color='#e41a1c', label='0507 MLP 输出')
        ax.axhline(clip_val, color='#ff7f00', linewidth=0.6,
                   linestyle=':', alpha=0.8, label=f'clip ±{clip_val:.4f}')
        ax.axhline(-clip_val, color='#ff7f00', linewidth=0.6,
                   linestyle=':', alpha=0.8)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(5, color='gray', linewidth=0.5, linestyle='--',
                   alpha=0.6, label='5 kph')
        ax.axvline(25, color='gray', linewidth=0.5, linestyle='--',
                   alpha=0.6, label='25 kph')
        ax.set_xlabel('vx_t (kph)'); ax.set_ylabel(f'{OUTPUT_NAMES[i]}')
        ax.set_title(f'输出分量 {i}: {OUTPUT_NAMES[i]}')
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'mlp_static_vx_scan.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  saved: {out_path}")


def main():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'diagnostic', 'mlp_instability')
    print(f'数据目录：{base}')

    scenarios = [
        ('straight_5kph', '直行 5 kph (30 m)'),
        ('circle_25kph_R80', '稳态圆周 25 kph (R=80 m)'),
        ('lane_change_5kph', '低速变道 5 kph'),
        ('clothoid_left_5kph', 'clothoid 左转 5 kph'),
    ]

    all_data = {}
    print('\n--- 加载数据 ---')
    for k, label in scenarios:
        d = load_scenario(base, k)
        all_data[k] = d
        print(f'  {k}: {len(d)} 变体')

    print('\n--- 单场景诊断面板 ---')
    for k, label in scenarios:
        plot_scenario_panel(k, label, all_data[k], base)

    print('\n--- 早期失控时序 ---')
    for k, label in scenarios:
        plot_early_steps(k, label, all_data[k], base, n_steps=200)

    print('\n--- 跨场景总览 ---')
    plot_cross_scenario_summary(all_data, base)

    print('\n--- MLP 静态扫描 ---')
    plot_mlp_static_test(base)

    print(f'\n所有图已存到：{base}')


if __name__ == '__main__':
    main()
