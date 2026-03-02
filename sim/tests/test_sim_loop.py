# sim/tests/test_sim_loop.py
"""闭环仿真测试。包含 V1 兼容测试 + differentiable 梯度测试。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim_loop import run_simulation
from config import load_config
from model.trajectory import generate_straight, generate_circle
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController


class TestSimLoop:
    def test_straight_line_tracks(self):
        """直线轨迹应能跟踪，最终横向误差很小。"""
        traj = generate_straight(length=200, speed=10.0)
        history = run_simulation(traj, init_speed=10.0)
        # 检查最后 2 秒的横向误差
        n_last = 100  # 2s at 50Hz
        for rec in history[-n_last:]:
            assert abs(rec['lateral_error']) < 1.0, \
                f"lateral_error={rec['lateral_error']:.3f} too large"

    def test_history_has_required_fields(self):
        traj = generate_straight(length=100, speed=5.0)
        history = run_simulation(traj, init_speed=5.0)
        assert len(history) > 0
        rec = history[0]
        for key in ['t', 'x', 'y', 'yaw', 'v', 'steer', 'acc',
                     'lateral_error', 'heading_error',
                     'ref_x', 'ref_y']:
            assert key in rec, f"Missing key: {key}"

    def test_circle_tracks(self):
        """圆弧轨迹应能跟踪（90度弧线）。
        注：LatControllerTruck 无积分器，长弧线会有累积偏移，
        因此使用 pi/2 弧线 + 较大半径测试跟踪能力。"""
        traj = generate_circle(radius=50.0, speed=5.0,
                               arc_angle=math.pi / 2)
        history = run_simulation(traj, init_speed=5.0)
        # 检查后半段横向误差
        n = len(history)
        for rec in history[n // 2:]:
            assert abs(rec['lateral_error']) < 5.0


class TestSimLoopDifferentiable:
    """differentiable=True 模式测试：梯度流 + 输出格式。"""

    def test_gradient_flows_through_lat(self):
        """横向误差 loss 应有梯度流向 lat_ctrl 参数（T2_y 表）。"""
        cfg = load_config()
        lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        lon_ctrl = LonController(cfg, differentiable=True)
        traj = generate_straight(length=50, speed=5.0)
        history = run_simulation(traj, init_speed=5.0, cfg=cfg,
                                 lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                 differentiable=True)
        lat_errs = torch.stack([h['lateral_error'] for h in history])
        loss = (lat_errs ** 2).mean()
        loss.backward()
        assert lat_ctrl.T2_y.grad is not None, "T2_y 应有梯度"

    def test_gradient_flows_through_lon(self):
        """速度跟踪 loss 应有梯度流向 lon_ctrl 参数。"""
        cfg = load_config()
        lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        lon_ctrl = LonController(cfg, differentiable=True)
        traj = generate_straight(length=50, speed=5.0)
        history = run_simulation(traj, init_speed=5.0, cfg=cfg,
                                 lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                 differentiable=True)
        # 速度误差 loss
        v_errs = torch.stack([h['v'] - 5.0 for h in history])
        loss = (v_errs ** 2).mean()
        loss.backward()
        assert lon_ctrl.low_speed_kp.grad is not None, "low_speed_kp 应有梯度"

    def test_history_values_are_tensors(self):
        """differentiable=True 时 history 中应为 tensor。"""
        cfg = load_config()
        lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        lon_ctrl = LonController(cfg, differentiable=True)
        traj = generate_straight(length=30, speed=5.0)
        history = run_simulation(traj, init_speed=5.0, cfg=cfg,
                                 lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                 differentiable=True)
        rec = history[10]
        for key in ['x', 'y', 'yaw', 'v', 'steer', 'acc',
                     'lateral_error', 'heading_error']:
            assert isinstance(rec[key], torch.Tensor), \
                f"history['{key}'] 应为 tensor，实际为 {type(rec[key])}"

    def test_false_matches_existing(self):
        """differentiable=False 应与现有 V1 行为一致（直线跟踪误差小）。"""
        traj = generate_straight(length=200, speed=10.0)
        history = run_simulation(traj, init_speed=10.0, differentiable=False)
        n_last = 100
        for rec in history[-n_last:]:
            val = rec['lateral_error']
            if isinstance(val, torch.Tensor):
                val = val.item()
            assert abs(val) < 1.0

    def test_external_controllers_reset(self):
        """外部传入控制器时应调用 reset_state()。"""
        cfg = load_config()
        lat_ctrl = LatControllerTruck(cfg, differentiable=False)
        lon_ctrl = LonController(cfg, differentiable=False)
        # 先跑一次，让状态不为零
        traj = generate_straight(length=50, speed=5.0)
        run_simulation(traj, init_speed=5.0, cfg=cfg,
                       lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                       differentiable=False)
        # 再跑一次，检查 reset 后首步状态为 0
        history = run_simulation(traj, init_speed=5.0, cfg=cfg,
                                 lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                 differentiable=False)
        # 首步转向应接近 0（直线跟踪，初始对齐）
        assert abs(history[0]['steer']) < 5.0

    def test_differentiable_straight_tracks(self):
        """differentiable=True 直线仿真也应能跟踪。"""
        cfg = load_config()
        lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        lon_ctrl = LonController(cfg, differentiable=True)
        traj = generate_straight(length=100, speed=5.0)
        history = run_simulation(traj, init_speed=5.0, cfg=cfg,
                                 lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                 differentiable=True)
        n_last = 50
        for rec in history[-n_last:]:
            val = rec['lateral_error']
            if isinstance(val, torch.Tensor):
                val = val.item()
            assert abs(val) < 2.0, \
                f"differentiable 直线跟踪横向误差过大: {val:.3f}"
