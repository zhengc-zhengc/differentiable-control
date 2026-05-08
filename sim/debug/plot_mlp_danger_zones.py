"""扫描 0507 MLP 输入空间，画"什么输入会让输出失控"。

策略：
1. 用最干净的"无挂车 + 无相对位姿差 + cos_rel_yaw=1"基线，挑 5 个核心输入
   维度（vx_t, vy_t, r_t, steer_sw, rear_torque），其余固定在训练均值。
2. 一维扫描：每个输入单独扫，画 0507 vs 默认 MLP 的输出曲线。
3. 二维扫描：扫两个核心输入的笛卡尔积，画输出量级 / 输出符号 的热图。
4. 把 4 个闭环场景的实际输入轨迹（vx, vy）叠在 (vx,vy) 热图上，
   直观看到"车体到底走到了哪、那里 MLP 在说什么"。
5. 比较 0507 / 0506 / 默认 MLP 的危险区差异。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(THIS_DIR)
os.chdir(SIM_DIR)
sys.path.insert(0, SIM_DIR)

from config import load_config, apply_plant_override
from model.vehicle_factory import create_vehicle

# 测试 MLP 标签（由 TEST_LABEL / TEST_CKPT_NAME 环境变量指定，兜底 0507）
TEST_LABEL = os.environ.get('TEST_LABEL', '0507')
TEST_CKPT_NAME = os.environ.get(
    'TEST_CKPT_NAME', 'best_truck_trailer_error_model_0507.pth')


# ===== MLP 加载 =====
def load_mlp(checkpoint_name):
    cfg = load_config('configs/train_with_0507.yaml')
    apply_plant_override(cfg, 'truck_trailer')
    cfg['truck_trailer_vehicle']['checkpoint_path'] = (
        f'configs/checkpoints/{checkpoint_name}')
    car = create_vehicle(cfg, x=0, y=0, yaw=0, v=0, dt=0.02,
                         differentiable=False)
    return (car._mlp, car._feature_mean.numpy().flatten(),
            car._feature_scale.numpy().flatten(),
            car._motion_error_clip.numpy().flatten())


# 14D 输入索引
IDX = {'trailer_mass': 0, 'has_trailer': 1, 'vx_t': 2, 'vy_t': 3, 'r_t': 4,
       'vx_s': 5, 'vy_s': 6, 'r_s': 7,
       'rel_x': 8, 'rel_y': 9, 'sin_rel_yaw': 10, 'cos_rel_yaw': 11,
       'steer_sw': 12, 'rear_torque': 13}


def make_baseline_input(n=1, has_trailer=False):
    """无挂车基线：trailer/has_trailer = 0，rel_x = -4.33（默认），
    cos_rel_yaw=1，其余 = 0。"""
    base = np.zeros((n, 14))
    base[:, IDX['cos_rel_yaw']] = 1.0
    base[:, IDX['rel_x']] = -4.331  # truckdynamicmodel 里的默认 hitch 偏移
    if has_trailer:
        base[:, IDX['trailer_mass']] = 15004.0
        base[:, IDX['has_trailer']] = 1.0
    return base


def forward(mlp, fm, fs, raw):
    norm = (raw - fm) / fs
    with torch.no_grad():
        out = mlp(torch.tensor(norm, dtype=torch.float32)).numpy()
    return out


# ===== 1D 扫描 =====
def sweep_1d(mlp, fm, fs, dim_name, values, base=None):
    """沿单一维扫描，其余维 = baseline。"""
    n = len(values)
    raw = base.copy() if base is not None else make_baseline_input(n)
    if base is not None and base.shape[0] == 1:
        raw = np.tile(base, (n, 1))
    raw[:, IDX[dim_name]] = values
    out = forward(mlp, fm, fs, raw)
    return out


def plot_1d_sweeps(mlps_dict, out_dir):
    """画 5 个核心输入维的 1D 扫描，对比 3 个 MLP。"""
    fig, axes = plt.subplots(5, 3, figsize=(16, 18))
    fig.suptitle('0507 / 0506 / 默认 MLP 一维输入扫描：'
                 '哪些输入会让输出偏离零',
                 fontsize=15, fontweight='bold')

    sweeps = [
        ('vx_t', np.linspace(0.0, 16.0, 200), '牵引车纵向速度 vx_t (m/s)',
         'vx 0~57 kph，其余 = 训练均值'),
        ('vy_t', np.linspace(-1.0, 1.0, 200), '牵引车侧向速度 vy_t (m/s)',
         '侧滑速度 ±1 m/s，vx 设 5 kph'),
        ('r_t', np.linspace(-0.3, 0.3, 200), '牵引车横摆角速度 r_t (rad/s)',
         '横摆角速度 ±0.3 rad/s，vx 设 5 kph'),
        ('steer_sw', np.linspace(-1.5, 1.5, 200), '方向盘角 (rad)',
         '方向盘 ±86°，vx 设 5 kph'),
        ('rear_torque', np.linspace(-3000, 5000, 200),
         '后驱车轮总扭矩 (N·m)', '后扭矩 ±5000 N·m，vx 设 5 kph'),
    ]

    out_dim_names = ['vx_t (m/s)', 'vy_t (m/s)', 'r_t (rad/s)']
    out_dim_idx = [0, 1, 2]
    colors = {TEST_LABEL: '#e41a1c', '0506': '#984ea3', '默认': '#377eb8'}

    for row, (dim_name, values, xlabel, sub) in enumerate(sweeps):
        # 准备 baseline：vy/r/steer/torque sweep 时把 vx 设为 5 kph
        base = make_baseline_input(1)
        if dim_name != 'vx_t':
            base[0, IDX['vx_t']] = 5.0 / 3.6  # 5 kph
            base[0, IDX['vx_s']] = 5.0 / 3.6

        for col, (out_idx, out_name) in enumerate(zip(out_dim_idx,
                                                       out_dim_names)):
            ax = axes[row, col]
            for tag, (mlp, fm, fs, clip) in mlps_dict.items():
                out = sweep_1d(mlp, fm, fs, dim_name, values, base=base)
                ax.plot(values, out[:, out_idx], '-', linewidth=1.6,
                        color=colors[tag], label=f'{tag} MLP', alpha=0.9)
                ax.axhline(clip[out_idx], linestyle=':', linewidth=0.5,
                           color=colors[tag], alpha=0.5)
                ax.axhline(-clip[out_idx], linestyle=':', linewidth=0.5,
                           color=colors[tag], alpha=0.5)
            ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
            ax.set_xlabel(xlabel + f'\n（{sub}）', fontsize=9)
            ax.set_ylabel(f'{out_name}', fontsize=10)
            if row == 0:
                ax.set_title(f'输出：{out_name}', fontsize=11,
                             fontweight='bold')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'danger_1d_sweeps.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


# ===== 2D 扫描 =====
def sweep_2d(mlp, fm, fs, x_dim, x_vals, y_dim, y_vals, base):
    """二维笛卡尔积扫描。返回 shape=(len(y), len(x), 9) 的输出张量。"""
    X, Y = np.meshgrid(x_vals, y_vals)
    flat = base[0:1].repeat(X.size, axis=0)
    flat[:, IDX[x_dim]] = X.flatten()
    flat[:, IDX[y_dim]] = Y.flatten()
    # 注意：vx_s 与 vx_t 联动；vy_s 与 vy_t 联动（在静态测试中保持）
    if x_dim == 'vx_t':
        flat[:, IDX['vx_s']] = X.flatten()
    if y_dim == 'vx_t':
        flat[:, IDX['vx_s']] = Y.flatten()
    if x_dim == 'vy_t':
        flat[:, IDX['vy_s']] = X.flatten()
    if y_dim == 'vy_t':
        flat[:, IDX['vy_s']] = Y.flatten()
    out = forward(mlp, fm, fs, flat)
    return out.reshape(X.shape[0], X.shape[1], 9), X, Y


def plot_2d_with_overlay(mlps_dict, scenarios_data, out_dir):
    """二维 (vx_t, vy_t) 扫描，把闭环轨迹叠在上面。"""
    vx_grid = np.linspace(0.0, 16.0, 121)  # 0~57 kph
    vy_grid = np.linspace(-0.6, 0.6, 121)

    base = make_baseline_input(1)

    # 输出三分量：vx_t / vy_t / r_t
    out_specs = [
        (0, 'MLP 输出 vx_t (m/s)', 0.15),
        (1, 'MLP 输出 vy_t (m/s)', 0.5),
        (2, 'MLP 输出 r_t (rad/s)', 0.008),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('二维输入扫描：(vx_t, vy_t) 平面 → MLP 输出三分量\n'
                 '(其余特征 = 训练均值/默认；overlay = 4 个闭环场景实际走过的输入点)',
                 fontsize=14, fontweight='bold')

    mlp_tags = ['默认', '0506', TEST_LABEL]

    for col, tag in enumerate(mlp_tags):
        mlp, fm, fs, clip = mlps_dict[tag]
        out_grid, X, Y = sweep_2d(mlp, fm, fs, 'vx_t', vx_grid,
                                   'vy_t', vy_grid, base)
        for row, (oi, oname, vmax) in enumerate(out_specs):
            ax = axes[row, col]
            data = out_grid[:, :, oi]
            # 用对称 diverging
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(data, extent=[vx_grid[0]*3.6, vx_grid[-1]*3.6,
                                         vy_grid[0], vy_grid[-1]],
                           origin='lower', aspect='auto', cmap='RdBu_r',
                           norm=norm)
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label(oname, fontsize=9)

            # overlay: 闭环里实际走过的 (vx_t, vy_t) 状态点
            scenario_colors = {
                'straight_5kph': '#222222',
                'circle_25kph_R80': '#ff7f00',
                'lane_change_5kph': '#4daf4a',
                'clothoid_left_5kph': '#e41a1c',
            }
            scenario_labels_zh = {
                'straight_5kph': '直行 5kph',
                'circle_25kph_R80': '圆周 25kph',
                'lane_change_5kph': '变道 5kph',
                'clothoid_left_5kph': 'clothoid 5kph',
            }
            for s, d in scenarios_data.items():
                # state[3] = vx_t, state[4] = vy_t
                state = d['state_after_mlp']
                vx = state[:, 3] * 3.6  # m/s -> kph for x
                vy = state[:, 4]
                ax.scatter(vx, vy, s=2, color=scenario_colors[s],
                           alpha=0.5,
                           label=scenario_labels_zh[s] if (col == 0 and row == 0) else None)
            ax.axhline(0, color='gray', linewidth=0.4, alpha=0.5)
            ax.axvline(5, color='gray', linewidth=0.4, linestyle=':',
                       alpha=0.6)
            ax.axvline(25, color='gray', linewidth=0.4, linestyle=':',
                       alpha=0.6)
            ax.set_xlabel('vx_t (kph)'); ax.set_ylabel('vy_t (m/s)')
            ax.set_title(f'{tag} MLP — {oname}', fontsize=11,
                         fontweight='bold')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right',
                          markerscale=3.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'danger_2d_vx_vy.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_2d_vy_r(mlps_dict, scenarios_data, out_dir):
    """二维 (vy_t, r_t) 扫描，固定 vx=5kph。"""
    vy_grid = np.linspace(-0.6, 0.6, 121)
    r_grid = np.linspace(-0.15, 0.15, 121)

    base = make_baseline_input(1)
    base[0, IDX['vx_t']] = 5.0 / 3.6
    base[0, IDX['vx_s']] = 5.0 / 3.6

    out_specs = [
        (0, 'MLP 输出 vx_t (m/s)', 0.15),
        (1, 'MLP 输出 vy_t (m/s)', 0.5),
        (2, 'MLP 输出 r_t (rad/s)', 0.008),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('二维输入扫描：(vy_t, r_t) 平面 → MLP 输出三分量\n'
                 '(vx_t = 5 kph 固定；overlay = 闭环里实际走过的 (vy_t, r_t))',
                 fontsize=14, fontweight='bold')

    mlp_tags = ['默认', '0506', TEST_LABEL]

    for col, tag in enumerate(mlp_tags):
        mlp, fm, fs, clip = mlps_dict[tag]
        out_grid, X, Y = sweep_2d(mlp, fm, fs, 'vy_t', vy_grid,
                                   'r_t', r_grid, base)
        for row, (oi, oname, vmax) in enumerate(out_specs):
            ax = axes[row, col]
            data = out_grid[:, :, oi]
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(data, extent=[vy_grid[0], vy_grid[-1],
                                         r_grid[0], r_grid[-1]],
                           origin='lower', aspect='auto', cmap='RdBu_r',
                           norm=norm)
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label(oname, fontsize=9)

            # overlay
            scenario_colors = {
                'lane_change_5kph': '#4daf4a',
                'clothoid_left_5kph': '#e41a1c',
            }
            scenario_labels_zh = {
                'lane_change_5kph': '变道 5kph',
                'clothoid_left_5kph': 'clothoid 5kph',
            }
            for s, d in scenarios_data.items():
                if s not in scenario_colors:
                    continue
                state = d['state_after_mlp']
                vy = state[:, 4]
                r = state[:, 5]
                ax.scatter(vy, r, s=2, color=scenario_colors[s],
                           alpha=0.5,
                           label=scenario_labels_zh[s] if (col == 0 and row == 0) else None)
            ax.axhline(0, color='gray', linewidth=0.4, alpha=0.5)
            ax.axvline(0, color='gray', linewidth=0.4, alpha=0.5)
            ax.set_xlabel('vy_t (m/s)'); ax.set_ylabel('r_t (rad/s)')
            ax.set_title(f'{tag} MLP — {oname}', fontsize=11,
                         fontweight='bold')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right',
                          markerscale=3.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'danger_2d_vy_r.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


# ===== 闭环输入散点 + 输出量级时序 =====
def plot_input_to_output_correlation(scenarios_data, out_dir,
                                      mlp_tag=None):
    if mlp_tag is None:
        mlp_tag = TEST_LABEL
    """对每个场景画 (输入特征 vs 输出量级) 散点 + 输出时序的"加权"图。"""
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'{mlp_tag} MLP 闭环：实际输入 vs MLP 输出关系（4 个场景）',
                 fontsize=14, fontweight='bold')

    scen_labels = {
        'straight_5kph': '直行 5kph',
        'circle_25kph_R80': '圆周 25kph (R=80m)',
        'lane_change_5kph': '变道 5kph',
        'clothoid_left_5kph': 'clothoid 左转 5kph',
    }

    for row, (s, label) in enumerate(scen_labels.items()):
        if s not in scenarios_data:
            continue
        d = scenarios_data[s]
        out = d['mlp_output_clipped']
        state = d['state_after_mlp']
        n = min(out.shape[0], state.shape[0])
        t = np.arange(n) * 0.02

        # 列 0: 输出量级时序（vy_t / r_t）
        ax = axes[row, 0]
        ax.plot(t, out[:n, 1], '-', linewidth=0.7, color='#e41a1c',
                label='vy_t 残差', alpha=0.85)
        ax.plot(t, out[:n, 2] * 50, '-', linewidth=0.7, color='#4daf4a',
                label='r_t 残差 ×50', alpha=0.85)
        ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('MLP 输出')
        ax.set_title(f'{label} — 输出时序')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # 列 1: 输入 vy_t vs 输出 vy_t（散点上色 = 时间）
        ax = axes[row, 1]
        sc = ax.scatter(state[:n, 4], out[:n, 1], c=t, s=4, cmap='viridis',
                        alpha=0.6)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('时间 (s)', fontsize=8)
        ax.axhline(0, color='gray', linewidth=0.4)
        ax.axvline(0, color='gray', linewidth=0.4)
        ax.set_xlabel('实际 vy_t (m/s)')
        ax.set_ylabel('MLP vy_t 残差输出 (m/s)')
        ax.set_title(f'{label} — 输入 vy_t vs 输出 vy_t')
        ax.grid(True, alpha=0.3)

        # 列 2: 输入 r_t vs 输出 r_t
        ax = axes[row, 2]
        sc = ax.scatter(state[:n, 5], out[:n, 2], c=t, s=4, cmap='viridis',
                        alpha=0.6)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('时间 (s)', fontsize=8)
        ax.axhline(0, color='gray', linewidth=0.4)
        ax.axvline(0, color='gray', linewidth=0.4)
        ax.set_xlabel('实际 r_t (rad/s)')
        ax.set_ylabel('MLP r_t 残差输出 (rad/s)')
        ax.set_title(f'{label} — 输入 r_t vs 输出 r_t')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'danger_input_output_correlation.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    out_dir = os.path.join(SIM_DIR, 'results', 'diagnostic',
                           'mlp_instability')

    print(f'--- 加载 3 个 MLP（测试 = {TEST_LABEL}）---')
    mlps_dict = {
        TEST_LABEL: load_mlp(TEST_CKPT_NAME),
        '0506': load_mlp('best_truck_trailer_error_model_0506.pth'),
        '默认': load_mlp('best_truck_trailer_error_model.pth'),
    }

    print('--- 加载闭环数据 ---')
    base = out_dir
    scenarios_data = {}
    for s in ['straight_5kph', 'circle_25kph_R80', 'lane_change_5kph',
              'clothoid_left_5kph']:
        path = os.path.join(base, s, 'mlp_test_full.npz')
        if os.path.exists(path):
            scenarios_data[s] = dict(np.load(path, allow_pickle=True))

    print('\n--- 1D 扫描 ---')
    plot_1d_sweeps(mlps_dict, out_dir)

    print('\n--- 2D 扫描 (vx, vy) ---')
    plot_2d_with_overlay(mlps_dict, scenarios_data, out_dir)

    print('\n--- 2D 扫描 (vy, r) @ vx=5kph ---')
    plot_2d_vy_r(mlps_dict, scenarios_data, out_dir)

    print('\n--- 闭环输入 vs 输出相关性 ---')
    plot_input_to_output_correlation(scenarios_data, out_dir)

    print(f'\n所有危险区图已存：{out_dir}')


if __name__ == '__main__':
    main()
