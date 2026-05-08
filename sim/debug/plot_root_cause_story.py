"""把 0507 MLP 失控原因讲成一张图。

故事线：
  Step 1（开环静态扫描）：在最干净的输入下（直行、无控制、无侧向），0507 MLP
  对 vy_t（牵引车侧向速度）输出一条 **随车速线性增长的正向偏置**。这条偏置在
  5 kph 时 ≈ 0.001 m/s（看起来无害），在 25 kph 时 ≈ 0.07 m/s。
  默认 MLP 在同样输入下偏置很小、近零。

  Step 2（闭环单场景）：把这条偏置每个 50Hz 步加进底层动力学，几百毫秒后
  实际 vy_t 已经被 MLP 拽偏，控制器开始反向打方向盘补偿。

  Step 3（积分爆炸）：方向盘动起来后状态进入 MLP 训练分布外的区域，MLP 对
  vy_t / r_t 的残差量级跳到 ±0.5 m/s 级别（撞 clip），方向盘命令也跟着发疯。

  Step 4（消融验证）：把 MLP 的牵引车速度三分量（vx_t / vy_t / r_t）置零，
  闭环立刻和"无 MLP 纯 RK4"完全重合。这证明问题集中在 MLP 输出的 9 个分量
  里前 3 个上，跟相对位姿残差无关。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch

# 切到 sim/ 目录方便载配置
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(THIS_DIR)
os.chdir(SIM_DIR)
sys.path.insert(0, SIM_DIR)

from config import load_config, apply_plant_override
from model.vehicle_factory import create_vehicle


def load_mlp_from_cfg(cfg):
    car = create_vehicle(cfg, x=0, y=0, yaw=0, v=0, dt=0.02,
                         differentiable=False)
    return car._mlp, car._feature_mean.numpy().flatten(), \
        car._feature_scale.numpy().flatten(), \
        car._motion_error_clip.numpy().flatten()


def get_mlp(checkpoint_name='best_truck_trailer_error_model_0507.pth'):
    cfg = load_config('configs/train_with_0507.yaml')
    apply_plant_override(cfg, 'truck_trailer')
    cfg['truck_trailer_vehicle']['checkpoint_path'] = (
        f'configs/checkpoints/{checkpoint_name}')
    return load_mlp_from_cfg(cfg)


def static_scan_vy_vs_speed(mlp, fm, fs, clip):
    """在最干净状态下（vy=r=0, 无控制, 无挂车）扫描 vx，记录 9D 输出。"""
    vx_kph = np.linspace(0.1, 60, 200)
    vx = vx_kph / 3.6
    raw = np.stack([
        np.zeros_like(vx),  # trailer_mass
        np.zeros_like(vx),  # has_trailer
        vx,                 # vx_t
        np.zeros_like(vx),  # vy_t
        np.zeros_like(vx),  # r_t
        vx,                 # vx_s
        np.zeros_like(vx),  # vy_s
        np.zeros_like(vx),  # r_s
        np.zeros_like(vx),  # rel_x
        np.zeros_like(vx),  # rel_y
        np.zeros_like(vx),  # sin_rel_yaw
        np.ones_like(vx),   # cos_rel_yaw
        np.zeros_like(vx),  # steer_sw
        np.zeros_like(vx),  # rear_torque
    ], axis=1)
    norm = (raw - fm) / fs
    with torch.no_grad():
        out = mlp(torch.tensor(norm, dtype=torch.float32)).numpy()
    return vx_kph, out, np.clip(out, -clip, clip)


def main():
    out_dir = os.path.join(SIM_DIR, 'results', 'diagnostic',
                           'mlp_instability')
    os.makedirs(out_dir, exist_ok=True)

    test_ckpt_name = os.environ.get(
        'TEST_CKPT_NAME', 'best_truck_trailer_error_model_0507.pth')
    print(f'--- 加载 MLP（测试 ckpt = {test_ckpt_name}）---')
    mlp_0507, fm_0507, fs_0507, clip_0507 = get_mlp(test_ckpt_name)
    mlp_def, fm_def, fs_def, clip_def = get_mlp(
        'best_truck_trailer_error_model.pth')
    mlp_0506, fm_0506, fs_0506, clip_0506 = get_mlp(
        'best_truck_trailer_error_model_0506.pth')

    print('--- 静态扫描 ---')
    vx_kph, raw_0507, clip_o_0507 = static_scan_vy_vs_speed(
        mlp_0507, fm_0507, fs_0507, clip_0507)
    _, raw_def, clip_o_def = static_scan_vy_vs_speed(
        mlp_def, fm_def, fs_def, clip_def)
    _, raw_0506, clip_o_0506 = static_scan_vy_vs_speed(
        mlp_0506, fm_0506, fs_0506, clip_0506)

    # ---- 加载闭环数据（lane_change_5kph 作为代表）----
    base = os.path.join(SIM_DIR, 'results', 'diagnostic', 'mlp_instability')
    scen = 'lane_change_5kph'
    sd = os.path.join(base, scen)
    no = dict(np.load(os.path.join(sd, 'no_mlp.npz'), allow_pickle=True))
    full = dict(np.load(os.path.join(sd, 'mlp_test_full.npz'),
                        allow_pickle=True))
    zero_vel = dict(np.load(os.path.join(sd, 'mlp_test_zero_vel_t.npz'),
                            allow_pickle=True))
    n = min(len(no['hist_t']), len(full['hist_t']))

    # ====================== 大图：4 行 ======================
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle('0507 MLP 失控根因 — 一图说清',
                 fontsize=18, fontweight='bold', y=0.995)
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.30)

    # ----- 第 1 行：开环静态扫描 — 三个 MLP 对比 vy_t 输出随车速 -----
    ax = fig.add_subplot(gs[0, :])
    ax.plot(vx_kph, raw_0507[:, 1], '-', linewidth=2.4, color='#e41a1c',
            label='0507 MLP（出问题的）')
    ax.plot(vx_kph, raw_0506[:, 1], '-', linewidth=1.8, color='#984ea3',
            label='0506 MLP（同架构上一版）', alpha=0.8)
    ax.plot(vx_kph, raw_def[:, 1], '-', linewidth=2.0, color='#377eb8',
            label='默认 MLP（64 隐层、上线版本）')
    clip_vy = float(clip_0507[1])
    ax.fill_between(vx_kph, -clip_vy, clip_vy, alpha=0.06, color='red',
                    label=f'0507 输出 clip 区间 ±{clip_vy:.3f} m/s')
    ax.axhline(0, color='k', linewidth=0.6)
    for vk in [5, 25]:
        ax.axvline(vk, color='gray', linestyle=':', linewidth=0.6,
                   alpha=0.7)
        ax.text(vk + 0.3, 0.18, f'{vk} kph', fontsize=9, color='gray')
    ax.set_xlabel('牵引车纵向速度 vx_t (kph)')
    ax.set_ylabel('MLP 输出 vy_t 残差 (m/s)')
    ax.set_title('① 开环静态扫描：在最干净的"直行无控制无侧向"输入下，'
                 '0507 MLP 凭空给出与车速近线性的 vy_t 偏置，默认 MLP 几乎为零',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.20)

    # ----- 第 2 行：闭环 — 实际 MLP 输出 + 实际 vy_t 状态 -----
    out_full = full['mlp_output_clipped']
    t_mlp = np.arange(out_full.shape[0]) * 0.02
    state_full = full['state_after_mlp']
    state_no = no['state_after_mlp']
    n2 = min(len(t_mlp), len(state_full), len(state_no))

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_mlp[:n2], out_full[:n2, 0], '-', color='#377eb8',
            linewidth=0.8, label='vx_t 残差', alpha=0.8)
    ax.plot(t_mlp[:n2], out_full[:n2, 1], '-', color='#e41a1c',
            linewidth=0.8, label='vy_t 残差', alpha=0.9)
    ax.plot(t_mlp[:n2], out_full[:n2, 2] * 50, '-', color='#4daf4a',
            linewidth=0.8, label='r_t 残差 ×50', alpha=0.8)
    ax.axhline(0, color='k', linewidth=0.4)
    ax.axhspan(-0.539, -0.50, alpha=0.1, color='red')
    ax.axhspan(0.50, 0.539, alpha=0.1, color='red')
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('MLP 输出（m/s 或 rad/s）')
    ax.set_title('② 闭环：MLP 输出从微小偏置变成大幅震荡')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_mlp[:n2], state_full[:n2, 4], '-', color='#e41a1c',
            linewidth=1.0, label='0507 闭环：实际 vy_t')
    ax.plot(t_mlp[:n2], state_no[:n2, 4], '-', color='#000000',
            linewidth=1.0, label='无 MLP：实际 vy_t', alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.4)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('车体侧向速度 vy_t (m/s)')
    ax.set_title('③ 实际 vy_t 状态：0507 把 vy 拉离 0 引发车体侧滑')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(full['hist_t'][:n], full['hist_steer'][:n], '-',
            color='#e41a1c', linewidth=0.9, label='0507 转向命令')
    ax.plot(no['hist_t'][:n], no['hist_steer'][:n], '-',
            color='#000000', linewidth=0.9, label='无 MLP 转向命令', alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.4)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('方向盘命令 (deg)')
    ax.set_title('④ 控制器反应：方向盘开始疯狂补偿（feedback runaway）')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ----- 第 3 行：偏离时序 + 早期累积 + 轨迹平面 -----
    ax = fig.add_subplot(gs[2, 0])
    dx = full['hist_x'][:n] - no['hist_x'][:n]
    dy = full['hist_y'][:n] - no['hist_y'][:n]
    dist = np.sqrt(dx**2 + dy**2)
    ax.plot(no['hist_t'][:n], dist, '-', linewidth=1.4, color='#e41a1c')
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('偏离基线距离 (m)')
    ax.set_yscale('log')
    ax.set_title('⑤ 与"无 MLP"基线偏差（log 轴）：1mm→1cm→1m 一路涨')
    ax.grid(True, alpha=0.3, which='both')
    # 标注关键时间点
    for thresh, color, label in [(0.001, '#aaaaaa', '1mm'),
                                  (0.01, '#888888', '1cm'),
                                  (0.1, '#666666', '10cm'),
                                  (1.0, '#444444', '1m')]:
        if np.any(dist > thresh):
            tt = no['hist_t'][np.argmax(dist > thresh)]
            ax.axvline(tt, color=color, linestyle=':', linewidth=0.6,
                       alpha=0.7)
            ax.text(tt + 0.3, thresh, f'{label}@t={tt:.1f}s',
                    fontsize=8, color=color)

    # 早期前 5 秒放大
    ax = fig.add_subplot(gs[2, 1])
    n_early = min(250, n2)
    t_early = no['hist_t'][:n_early]
    cum_vy_inj = np.cumsum(out_full[:n_early, 1]) * 0.02  # 累积 vy 注入 × dt
    ax.plot(t_early, dy[:n_early], '-', linewidth=1.4, color='#e41a1c',
            label='实际 y 偏差（0507 - no_mlp）')
    ax.plot(t_early, cum_vy_inj, '--', linewidth=1.2, color='#377eb8',
            label='∫(vy_t MLP 注入) dt   理论横向漂移', alpha=0.85)
    ax.axhline(0, color='k', linewidth=0.4)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('累积侧向漂移 (m)')
    ax.set_title('⑥ 早期 5 秒：实际偏差 ≈ MLP vy_t 注入的时间积分'
                 '\n（证明是"小偏置 × 大量步数"的累积过程，不是单步爆炸）')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 轨迹平面（早期 100 步）
    ax = fig.add_subplot(gs[2, 2])
    n_plot = min(500, n)
    ax.plot(no['ref_traj_x'][:n_plot], no['ref_traj_y'][:n_plot], 'k--',
            linewidth=1.0, alpha=0.6, label='参考轨迹')
    ax.plot(no['hist_x'][:n_plot], no['hist_y'][:n_plot], '-',
            color='#000000', linewidth=1.5, alpha=0.8, label='无 MLP')
    ax.plot(zero_vel['hist_x'][:n_plot], zero_vel['hist_y'][:n_plot], '-',
            color='#4daf4a', linewidth=1.5, alpha=0.8,
            label='0507 但置零牵引车 v 残差')
    ax.plot(full['hist_x'][:n_plot], full['hist_y'][:n_plot], '-',
            color='#e41a1c', linewidth=1.5, alpha=0.8, label='0507 完整')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('⑦ 轨迹平面（前 10 s）：0507 立刻偏离参考路径')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ----- 第 4 行：消融柱状图 -----
    ax = fig.add_subplot(gs[3, :])
    scenarios = ['straight_5kph', 'circle_25kph_R80', 'lane_change_5kph',
                 'clothoid_left_5kph']
    scen_labels = ['直行 5kph', '圆周 25kph (R=80)', '变道 5kph',
                   'clothoid 左转 5kph']
    variants = ['no_mlp', 'mlp_default', 'mlp_test_full',
                'mlp_test_zero_vel_t']
    var_labels = ['无 MLP（纯 RK4）', '默认 MLP', '0507 MLP 完整',
                  '0507 MLP 但牵引车 v 残差置零']
    var_colors = ['#000000', '#377eb8', '#e41a1c', '#4daf4a']

    rmse = {v: [] for v in variants}
    for s in scenarios:
        for v in variants:
            d = dict(np.load(os.path.join(base, s, f'{v}.npz'),
                             allow_pickle=True))
            rmse[v].append(
                float(np.sqrt(np.mean(d['hist_lat_err'] ** 2))))

    x = np.arange(len(scenarios))
    width = 0.20
    for i, v in enumerate(variants):
        offset = (i - 1.5) * width
        ax.bar(x + offset, rmse[v], width, color=var_colors[i],
               edgecolor='black', linewidth=0.4, label=var_labels[i])
        for xi, val in zip(x + offset, rmse[v]):
            if val > 0:
                ax.text(xi, val * 1.1, f'{val:.3f}', ha='center',
                        fontsize=8, color=var_colors[i])

    ax.set_xticks(x); ax.set_xticklabels(scen_labels, fontsize=11)
    ax.set_ylabel('横向 RMSE (m)（log 轴）')
    ax.set_yscale('log')
    ax.set_title('⑧ 组件消融验证：把 0507 MLP 的牵引车速度三分量 (vx_t, vy_t, r_t) '
                 '置零，立刻和"无 MLP"完全重合\n'
                 '→ 失控仅由前 3 个输出分量驱动，相对位姿残差不参与（无挂车模式被掩码）',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, ncol=4, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', which='both')

    out_path = os.path.join(out_dir, 'ROOT_CAUSE_STORY.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"保存：{out_path}")


if __name__ == '__main__':
    main()
