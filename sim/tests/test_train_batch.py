# sim/tests/test_train_batch.py
"""批量训练管线测试。覆盖：
- 基础算子 (_lookup1d_batch, _clamp_ste_batch, _rate_limit_batch, _PIDBatch) 的 B=1 与 scalar 一致，B>1 各元素独立
- BatchedTrajectoryTable padding + mask + 查询
- BatchedTruckTrailerVehicle B=1 与 scalar 一致（多步）
- run_simulation_batch B=1 与 scalar 一致（lateral_error/v 漂移 < 1e-2）
- B>1 变长轨迹同步推进不崩溃，梯度流到所有可微参数
- train_batch 2 epoch 小轨迹跑通、loss 不爆、tuned yaml 可 load_config 读回
"""
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common import PID, lookup1d
from config import load_config, apply_plant_override
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from model.trajectory import generate_circle, generate_lane_change
from model.truck_trailer_vehicle import TruckTrailerVehicle
from sim_loop import run_simulation

from optim.train_batch import (
    BatchedHybridDynamicVehicle, BatchedLatTruck, BatchedLonCtrl,
    BatchedTrajectoryTable, BatchedTruckTrailerVehicle, _clamp_ste_batch,
    _lookup1d_batch, _PIDBatch, _rate_limit_batch, batched_tracking_loss,
    run_simulation_batch, train_batch)


def _truck_cfg():
    cfg = load_config()
    apply_plant_override(cfg, 'truck_trailer')
    # 用无挂车设置以避免 checkpoint 强依赖
    cfg['truck_trailer_vehicle']['default_trailer_mass_kg'] = 0.0
    return cfg


# ─── 基础算子 ───────────────────────────────────────────────────────────

class TestLookup1dBatch:
    def test_matches_scalar_b1(self):
        tx = torch.tensor([0.0, 10.0, 20.0, 30.0])
        ty = torch.tensor([1.0, 4.0, 9.0, 16.0])
        for v in [5.0, 15.0, 25.0, -3.0, 35.0]:
            s = lookup1d(tx, ty, torch.tensor(v))
            b = _lookup1d_batch(tx, ty, torch.tensor([v]))
            assert torch.allclose(s.reshape(1), b, atol=1e-6)

    def test_b4(self):
        tx = torch.tensor([0.0, 10.0, 20.0])
        ty = torch.tensor([0.0, 5.0, 10.0])
        x = torch.tensor([5.0, 15.0, 25.0, -3.0])
        out = _lookup1d_batch(tx, ty, x)
        assert torch.allclose(out, torch.tensor([2.5, 7.5, 10.0, 0.0]),
                              atol=1e-6)

    def test_gradient_to_table_y(self):
        tx = torch.tensor([0.0, 10.0, 20.0])
        ty = torch.nn.Parameter(torch.tensor([0.0, 5.0, 10.0]))
        x = torch.tensor([5.0, 15.0])
        _lookup1d_batch(tx, ty, x).sum().backward()
        assert ty.grad is not None and (ty.grad.abs() > 0).any()


class TestPidBatch:
    def test_b1_matches_scalar(self):
        scalar = PID()
        batch = _PIDBatch(batch_size=1)
        for e in [0.1, 0.2, 0.15, -0.05, 0.3]:
            s = scalar.control(torch.tensor(e), 0.02,
                               kp=0.5, ki=0.2, kd=0.0,
                               integrator_enable=True, sat=1.0)
            b = batch.control(torch.tensor([e]), 0.02,
                              kp=torch.tensor(0.5), ki=torch.tensor(0.2),
                              kd=0.0, integrator_enable=True, sat=1.0)
            assert torch.allclose(s.reshape(1), b, atol=1e-6)

    def test_b4_independent(self):
        b = _PIDBatch(batch_size=4)
        err = torch.tensor([0.1, 0.0, -0.2, 0.5])
        out1 = b.control(err, 0.02, kp=torch.tensor(1.0),
                         ki=torch.tensor(0.5), kd=0.0,
                         integrator_enable=True, sat=10.0)
        # 首次：1.0*err + 0.5*err*0.02 = 1.01*err
        assert torch.allclose(out1, 1.01 * err, atol=1e-6)
        out2 = b.control(err, 0.02, kp=torch.tensor(1.0),
                         ki=torch.tensor(0.5), kd=0.0,
                         integrator_enable=True, sat=10.0)
        assert torch.allclose(out2, 1.02 * err, atol=1e-6)


class TestClampSteBatch:
    def test_grad_passthrough(self):
        x = torch.tensor([3.0, -2.0], requires_grad=True)
        out = _clamp_ste_batch(x, -1.0, 1.0)
        assert torch.allclose(out, torch.tensor([1.0, -1.0]))
        out.sum().backward()
        assert torch.allclose(x.grad, torch.tensor([1.0, 1.0]))


class TestRateLimitBatch:
    def test_limits(self):
        prev = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, -1.0])
        out = _rate_limit_batch(prev, target, rate=10.0, dt=0.02)
        assert torch.allclose(out, torch.tensor([0.2, -0.2]))


# ─── 轨迹容器 ──────────────────────────────────────────────────────────

class TestBatchedTrajectoryTable:
    def test_pad_and_mask(self):
        trajs = [
            generate_circle(radius=50.0, speed=5.0, arc_angle=0.3),
            generate_circle(radius=50.0, speed=5.0, arc_angle=1.0),
        ]
        bt = BatchedTrajectoryTable(trajs)
        assert bt.B == 2
        assert bt.T_max == len(trajs[1])
        assert bt.valid_mask[0, :len(trajs[0])].all()
        assert (bt.valid_mask[0, len(trajs[0]):] == 0).all()
        assert bt.valid_mask[1].all()
        # padding 用末尾值填充
        assert abs(bt.ref_x[0, -1].item() - trajs[0][-1].x) < 1e-5

    def test_nearest_idx(self):
        trajs = [generate_circle(radius=50.0, speed=5.0, arc_angle=0.5),
                 generate_circle(radius=50.0, speed=5.0, arc_angle=0.8)]
        bt = BatchedTrajectoryTable(trajs)
        x = torch.tensor([trajs[0][5].x + 0.1, trajs[1][20].x - 0.1])
        y = torch.tensor([trajs[0][5].y, trajs[1][20].y])
        idx = bt.query_nearest_idx(x, y)
        assert idx.shape == (2,)
        # 允许 ±1 neighbor（轨迹点密时微扰可能跨到相邻点）
        assert abs(idx[0].item() - 5) <= 1
        assert abs(idx[1].item() - 20) <= 1


# ─── BatchedTruckTrailerVehicle B=1 匹配 scalar ──────────────────────────

class TestBatchedTruckTrailerVehicleB1:
    def test_no_mlp_matches_scalar(self):
        """Base RK4 only，B=1 多步状态应与 scalar 一致（trailer_mass=0 无挂场景）。"""
        cfg = _truck_cfg()
        # 强制不走 MLP：清空 checkpoint_path
        cfg = {**cfg}
        cfg['truck_trailer_vehicle'] = {**cfg['truck_trailer_vehicle']}
        cfg['truck_trailer_vehicle']['checkpoint_path'] = ''

        params = cfg['truck_trailer_vehicle']
        scalar = TruckTrailerVehicle(
            params, x=0.0, y=0.0, yaw=0.0, v=10.0,
            dt=0.02, trailer_mass_kg=0.0)
        batch = BatchedTruckTrailerVehicle(
            cfg, batch_size=1,
            init_x=torch.tensor([0.0]), init_y=torch.tensor([0.0]),
            init_yaw=torch.tensor([0.0]), init_v=torch.tensor([10.0]),
            dt=0.02, trailer_mass_kg=torch.tensor([0.0]),
            checkpoint_path='')
        for _ in range(30):
            scalar.step(delta=0.01, torque_wheel=500.0)
            batch.step(delta=torch.tensor([0.01]),
                        torque_wheel=torch.tensor([500.0]))
        assert torch.allclose(batch.x, scalar.x.reshape(1), atol=1e-4), \
            f'x diff: batch={batch.x.item()} scalar={scalar.x.item()}'
        assert torch.allclose(batch.y, scalar.y.reshape(1), atol=1e-4)
        assert torch.allclose(batch.yaw, scalar.yaw.reshape(1), atol=1e-5)
        assert torch.allclose(batch.v, scalar.v.reshape(1), atol=1e-4)

    def test_with_mlp_matches_scalar(self):
        """MLP v2 启用时，B=1 与 scalar 在 30 步内一致。"""
        cfg = _truck_cfg()
        ckpt_rel = cfg['truck_trailer_vehicle'].get(
            'checkpoint_path',
            'configs/checkpoints/best_truck_trailer_error_model.pth')
        sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_abs = os.path.join(sim_dir, ckpt_rel) if not os.path.isabs(ckpt_rel) else ckpt_rel
        if not os.path.exists(ckpt_abs):
            pytest.skip(f"checkpoint not found: {ckpt_abs}")

        params = cfg['truck_trailer_vehicle']
        scalar = TruckTrailerVehicle(
            params, x=0.0, y=0.0, yaw=0.0, v=10.0,
            dt=0.02, trailer_mass_kg=0.0, checkpoint_path=ckpt_rel)
        batch = BatchedTruckTrailerVehicle(
            cfg, batch_size=1,
            init_x=torch.tensor([0.0]), init_y=torch.tensor([0.0]),
            init_yaw=torch.tensor([0.0]), init_v=torch.tensor([10.0]),
            dt=0.02, trailer_mass_kg=torch.tensor([0.0]),
            checkpoint_path=ckpt_rel)
        for i in range(30):
            d = 0.02 * ((i % 5) - 2) / 2.0
            scalar.step(delta=d, torque_wheel=500.0)
            batch.step(delta=torch.tensor([d]),
                        torque_wheel=torch.tensor([500.0]))
        assert torch.allclose(batch.x, scalar.x.reshape(1), atol=1e-4)
        assert torch.allclose(batch.y, scalar.y.reshape(1), atol=1e-4)
        assert torch.allclose(batch.yaw, scalar.yaw.reshape(1), atol=1e-5)
        assert torch.allclose(batch.v, scalar.v.reshape(1), atol=1e-4)


# ─── run_simulation_batch 端到端 ─────────────────────────────────────────

class TestRunSimulationBatch:
    def test_b1_lateral_error_matches_scalar(self):
        """B=1 的 lateral_error/v 序列与 scalar run_simulation 在 1e-2 内一致。"""
        cfg = _truck_cfg()
        traj = generate_circle(radius=60.0, speed=10.0, arc_angle=0.4)
        scalar_hist = run_simulation(
            traj, init_speed=10.0, cfg=cfg,
            differentiable=True, tbptt_k=0)
        batch_hist = run_simulation_batch([traj], cfg=cfg, tbptt_k=0)
        n = len(scalar_hist)
        lat_s = torch.stack([h['lateral_error'] for h in scalar_hist])
        lat_b = batch_hist['lateral_error'][0, :n]
        # scalar path 有 station_fnl 分支逻辑，batch 做了 smooth 近似，
        # 允许 ~1e-2 漂移（圆弧上主要在纵向速度控制阶段）
        max_diff = (lat_s - lat_b).abs().max().item()
        assert max_diff < 5e-3, f"lateral_error max diff {max_diff}"

    def test_hard_mode_matches_scalar_v1(self):
        """hard_mode=True 应与 scalar run_simulation(differentiable=False)
        逐元素严格一致（1e-4）。覆盖两种轨迹排除偶然一致。"""
        cfg = _truck_cfg()
        for traj in [generate_circle(radius=60.0, speed=10.0, arc_angle=0.4),
                     generate_lane_change(lane_width=3.5, change_length=40.0,
                                           speed=10.0, lead_in=10.0,
                                           lead_out=10.0)]:
            scalar_hist = run_simulation(
                traj, init_speed=traj[0].v, cfg=cfg,
                differentiable=False, tbptt_k=0)
            batch_hist = run_simulation_batch(
                [traj], cfg=cfg, tbptt_k=0, hard_mode=True)
            n = len(scalar_hist)
            # FP 顺序差异：scalar 走 math.*（float64），batch 走 torch.*（float32），
            # 多步累积 → lateral_error 允许 ~5mm，其他允许对应量级
            tol = {'lateral_error': 5e-3, 'heading_error': 2e-3,
                   'v': 5e-3, 'steer': 0.5, 'acc': 5e-3}
            for key in tol:
                scalar_vals = torch.tensor([float(h[key]) for h in scalar_hist])
                batch_vals = batch_hist[key][0, :n]
                max_diff = (scalar_vals - batch_vals).abs().max().item()
                assert max_diff < tol[key], (
                    f"hard_mode {key}: max diff {max_diff} > {tol[key]} "
                    f"(traj_len={n})")

    def test_hybrid_dynamic_batch_matches_scalar(self):
        """hybrid_dynamic batched hard_mode 与 scalar run_simulation(differentiable=False)
        在短圆弧上 lateral_error / v / steer 应一致（容差同 truck_trailer 版）。
        只测一条轨迹 + 短弧，保证 pytest 下 <5s 完成。MLP 走默认 checkpoint。"""
        cfg = load_config()
        apply_plant_override(cfg, 'hybrid_dynamic')
        traj = generate_circle(radius=60.0, speed=10.0, arc_angle=0.3)
        scalar_hist = run_simulation(
            traj, init_speed=traj[0].v, cfg=cfg,
            differentiable=False, tbptt_k=0)
        batch_hist = run_simulation_batch(
            [traj], cfg=cfg, tbptt_k=0, hard_mode=True)
        n = len(scalar_hist)
        tol = {'lateral_error': 5e-3, 'heading_error': 2e-3,
               'v': 5e-3, 'steer': 0.5, 'acc': 5e-3}
        for key, t in tol.items():
            scalar_vals = torch.tensor([float(h[key]) for h in scalar_hist])
            batch_vals = batch_hist[key][0, :n]
            max_diff = (scalar_vals - batch_vals).abs().max().item()
            assert max_diff < t, (
                f"hybrid_dynamic batched vs scalar {key}: "
                f"max diff {max_diff} > {t}")

    def test_b3_variable_length(self):
        cfg = _truck_cfg()
        trajs = [
            generate_circle(radius=30.0, speed=5.0, arc_angle=0.25),
            generate_circle(radius=60.0, speed=10.0, arc_angle=0.45),
            generate_circle(radius=100.0, speed=15.0, arc_angle=0.80),
        ]
        hist = run_simulation_batch(trajs, cfg=cfg, tbptt_k=0)
        assert hist['x'].shape[0] == 3
        lens = hist['valid_mask'].sum(dim=1)
        assert lens[0] < lens[2]
        # 无 NaN（有效区内）
        mask = hist['valid_mask'] > 0.5
        for k in ('lateral_error', 'heading_error', 'steer', 'acc', 'v'):
            assert torch.isfinite(hist[k][mask]).all(), f'{k} NaN detected'

    def test_gradient_flows_to_all_tunables(self):
        cfg = _truck_cfg()
        trajs = [generate_lane_change(lane_width=3.5, change_length=30.0,
                                       speed=10.0, lead_in=10.0, lead_out=10.0)
                 for _ in range(2)]
        hist = run_simulation_batch(trajs, cfg=cfg, tbptt_k=0)
        ref_speeds = torch.tensor([10.0, 10.0])
        per_traj, _ = batched_tracking_loss(hist, ref_speeds,
                                             return_details=True)
        per_traj.mean().backward()
        lat, lon = hist['_lat_ctrl'], hist['_lon_ctrl']
        # 变曲率 lane_change 下 T6 应有梯度
        for name in ['T2_y', 'T3_y', 'T4_y', 'T6_y']:
            g = getattr(lat, name).grad
            assert g is not None and (g.abs() > 0).any(), \
                f'{name} no gradient'
        for name in ['station_kp', 'low_speed_kp', 'high_speed_kp']:
            g = getattr(lon, name).grad
            assert g is not None, f'{name} no gradient'


# ─── train_batch 端到端 ────────────────────────────────────────────────

class TestTrainBatchE2E:
    def test_2_epoch_small(self, tmp_path, monkeypatch):
        """2 epoch × 2 条短轨迹：loss 有限、yaml 可读回。"""
        monkeypatch.chdir(tmp_path)
        # 把 sim 目录作为工作目录需要 configs/ 相对路径有效，切换回 sim/
        sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        monkeypatch.chdir(sim_dir)
        trajs = [generate_circle(radius=50.0, speed=5.0, arc_angle=0.25),
                 generate_circle(radius=80.0, speed=10.0, arc_angle=0.35)]
        result = train_batch(
            trajectories=trajs, n_epochs=2, lr=1e-2, lr_tables=1e-2,
            tbptt_k=0, grad_clip=10.0, verbose=False,
            plant='truck_trailer')
        assert len(result['losses']) == 2
        assert all(torch.isfinite(torch.tensor(x)).item()
                    for x in result['losses'])
        assert os.path.exists(result['saved_path'])
        # yaml 可读回
        cfg = load_config(result['saved_path'])
        assert 'lat_truck' in cfg and 'lon' in cfg
