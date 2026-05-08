"""把 0507 和 0508_train_loss 两个 ckpt 的诊断数据放一张图上对比。

预期：0508 把 0507 的"开环 vy_t 偏置随车速线性增长"问题修了大半，闭环
横向 RMSE 在变道/clothoid 上数量级缩小。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(THIS_DIR)
os.chdir(SIM_DIR)
sys.path.insert(0, SIM_DIR)

from config import load_config, apply_plant_override
from model.vehicle_factory import create_vehicle


def load_mlp(name):
    cfg = load_config('configs/train_with_0507.yaml')
    apply_plant_override(cfg, 'truck_trailer')
    cfg['truck_trailer_vehicle']['checkpoint_path'] = (
        f'configs/checkpoints/{name}')
    car = create_vehicle(cfg, x=0, y=0, yaw=0, v=0, dt=0.02,
                         differentiable=False)
    return (car._mlp, car._feature_mean.numpy().flatten(),
            car._feature_scale.numpy().flatten(),
            car._motion_error_clip.numpy().flatten())


def static_scan_vx(mlp, fm, fs):
    vx = np.linspace(0.0, 16.0, 200)
    raw = np.zeros((len(vx), 14))
    raw[:, 11] = 1.0   # cos_rel_yaw
    raw[:, 8] = -4.331  # rel_x
    raw[:, 2] = vx
    raw[:, 5] = vx
    norm = (raw - fm) / fs
    with torch.no_grad():
        out = mlp(torch.tensor(norm, dtype=torch.float32)).numpy()
    return vx * 3.6, out


def main():
    base = os.path.join(SIM_DIR, 'results', 'diagnostic', 'mlp_instability')
    print('--- 加载 MLP ---')
    mlps = {
        '默认': load_mlp('best_truck_trailer_error_model.pth'),
        '0507': load_mlp('best_truck_trailer_error_model_0507.pth'),
        '0508TL': load_mlp('best_truck_trailer_error_model_train_loss_0508.pth'),
    }
    colors = {'默认': '#377eb8', '0507': '#e41a1c', '0508TL': '#2ca02c'}

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('0507 vs 0508 train_loss MLP — 修复效果对比',
                 fontsize=16, fontweight='bold', y=0.995)
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.30)

    # ── 第 1 行：开环静态扫描，三条 vy_t 输出曲线 ──
    ax = fig.add_subplot(gs[0, :])
    for tag, (mlp, fm, fs, clip) in mlps.items():
        vx_kph, out = static_scan_vx(mlp, fm, fs)
        ax.plot(vx_kph, out[:, 1], '-', linewidth=2.2, color=colors[tag],
                label=f'{tag} MLP')
        ax.axhline(clip[1], linestyle=':', linewidth=0.6,
                   color=colors[tag], alpha=0.5)
        ax.axhline(-clip[1], linestyle=':', linewidth=0.6,
                   color=colors[tag], alpha=0.5)
    ax.axhline(0, color='k', linewidth=0.6)
    for vk in [5, 25]:
        ax.axvline(vk, color='gray', linestyle=':', linewidth=0.5,
                   alpha=0.6)
        ax.text(vk + 0.3, 0.16, f'{vk} kph', fontsize=8, color='gray')
    ax.set_xlabel('vx_t (kph)，其余输入 = 训练均值')
    ax.set_ylabel('MLP 输出 vy_t 残差 (m/s)')
    ax.set_title('① 开环静态扫描：0508 把 0507 的"vy_t 输出随车速线性增长"问题压平了',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.2)

    # ── 第 2 行：闭环每场景 lat_RMSE 对比 ──
    scenarios = ['straight_5kph', 'circle_25kph_R80', 'lane_change_5kph',
                 'clothoid_left_5kph']
    scen_labels = ['直行 5kph', '圆周 25kph (R=80)', '变道 5kph',
                   'clothoid 左转 5kph']
    variants = ['no_mlp', 'mlp_default', 'mlp_test_full',
                'mlp_test_zero_vel_t']
    var_labels = ['无 MLP', '默认 MLP', '完整', '零牵v']

    ckpts = ['0507', '0508_train_loss']
    ckpt_labels = ['0507', '0508TL']

    rmse = {ck: {v: [] for v in variants} for ck in ckpts}
    for ck in ckpts:
        for s in scenarios:
            for v in variants:
                d = dict(np.load(
                    os.path.join(base, ck, s, f'{v}.npz'),
                    allow_pickle=True))
                rmse[ck][v].append(
                    float(np.sqrt(np.mean(d['hist_lat_err'] ** 2))))

    ax = fig.add_subplot(gs[1, :])
    n_var = len(variants)
    width = 0.10
    n_groups = 2 * n_var  # 0507 + 0508 each variant
    offsets = np.linspace(-width * (n_groups - 1) / 2,
                          width * (n_groups - 1) / 2, n_groups)
    x = np.arange(len(scenarios))

    var_colors = ['#000000', '#377eb8', '#e41a1c', '#4daf4a']
    for vi, v in enumerate(variants):
        # 0507: 实色
        ax.bar(x + offsets[vi * 2], rmse['0507'][v], width,
               color=var_colors[vi], edgecolor='black', linewidth=0.4,
               label=f'0507 — {var_labels[vi]}')
        # 0508: 同色 hatch
        ax.bar(x + offsets[vi * 2 + 1], rmse['0508_train_loss'][v], width,
               color=var_colors[vi], hatch='///', alpha=0.8,
               edgecolor='black', linewidth=0.4,
               label=f'0508TL — {var_labels[vi]}')
    ax.set_xticks(x); ax.set_xticklabels(scen_labels, fontsize=10)
    ax.set_ylabel('横向 RMSE (m)（log 轴）')
    ax.set_yscale('log')
    ax.set_title('② 闭环 4 场景横向 RMSE：0508TL（hatch）vs 0507（实色）'
                 ' — 变道、clothoid 上 0508 几乎跟"无 MLP"重合',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=4, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', which='both')

    # ── 第 3 行：MLP 输出 RMS 量级对比（vy_t 残差为代表）──
    ax = fig.add_subplot(gs[2, :])
    width2 = 0.20
    offsets2 = np.linspace(-width2 / 2, width2 / 2, 2)
    component_idx = 1  # vy_t
    for ci, (ck, lbl) in enumerate(zip(ckpts, ckpt_labels)):
        rms_vy = []
        for s in scenarios:
            d = dict(np.load(
                os.path.join(base, ck, s, 'mlp_test_full.npz'),
                allow_pickle=True))
            out = d['mlp_output_clipped']
            rms_vy.append(float(np.sqrt(np.mean(
                out[:, component_idx] ** 2))))
        c = '#e41a1c' if ck == '0507' else '#2ca02c'
        ax.bar(x + offsets2[ci], rms_vy, width2, color=c,
               edgecolor='black', linewidth=0.5, label=f'{lbl} 完整')
        for xi, val in zip(x + offsets2[ci], rms_vy):
            ax.text(xi, val * 1.15, f'{val:.4f}', ha='center',
                    fontsize=8, color=c)
    ax.set_xticks(x); ax.set_xticklabels(scen_labels, fontsize=10)
    ax.set_ylabel('MLP vy_t 残差 RMS（m/s，log 轴）')
    ax.set_yscale('log')
    ax.set_title('③ 闭环里 MLP vy_t 残差输出量级：0508 普遍小一个数量级',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', which='both')

    # ── 第 4 行：闭环 vy_t 状态时序对比（lane_change & clothoid）──
    for col, (s, label) in enumerate(zip(['lane_change_5kph',
                                            'clothoid_left_5kph'],
                                           ['变道 5kph', 'clothoid 左转 5kph'])):
        ax = fig.add_subplot(gs[3, col])
        no = dict(np.load(os.path.join(base, '0507', s, 'no_mlp.npz'),
                          allow_pickle=True))
        full_07 = dict(np.load(os.path.join(base, '0507', s,
                                             'mlp_test_full.npz'),
                                 allow_pickle=True))
        full_08 = dict(np.load(os.path.join(base, '0508_train_loss', s,
                                             'mlp_test_full.npz'),
                                 allow_pickle=True))
        n07 = len(full_07['state_after_mlp'])
        n08 = len(full_08['state_after_mlp'])
        nm = len(no['state_after_mlp'])
        ax.plot(np.arange(nm) * 0.02, no['state_after_mlp'][:, 4],
                color='black', linewidth=1.0, label='无 MLP', alpha=0.7)
        ax.plot(np.arange(n07) * 0.02, full_07['state_after_mlp'][:, 4],
                color='#e41a1c', linewidth=1.0, label='0507 完整', alpha=0.85)
        ax.plot(np.arange(n08) * 0.02, full_08['state_after_mlp'][:, 4],
                color='#2ca02c', linewidth=1.0, label='0508TL 完整',
                alpha=0.85)
        ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('车体侧向速度 vy_t (m/s)')
        ax.set_title(f'④ {label} — 实际 vy_t 状态时序')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 第 4 行 col 2：方向盘命令（clothoid_left_5kph）
    ax = fig.add_subplot(gs[3, 2])
    s = 'clothoid_left_5kph'
    for ck, lbl, c in [('0507', '0507 完整', '#e41a1c'),
                        ('0508_train_loss', '0508TL 完整', '#2ca02c')]:
        d = dict(np.load(os.path.join(base, ck, s, 'mlp_test_full.npz'),
                         allow_pickle=True))
        ax.plot(d['hist_t'], d['hist_steer'], '-', linewidth=0.8,
                color=c, label=lbl, alpha=0.85)
    no = dict(np.load(os.path.join(base, '0507', s, 'no_mlp.npz'),
                      allow_pickle=True))
    ax.plot(no['hist_t'], no['hist_steer'], '-', linewidth=0.8,
            color='black', label='无 MLP', alpha=0.6)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('方向盘命令 (deg)')
    ax.set_title('⑤ clothoid 左转：方向盘命令（0507 疯狂震荡，0508 平稳）')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    out_path = os.path.join(base, '0507_vs_0508TL_comparison.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'保存：{out_path}')


if __name__ == '__main__':
    main()
