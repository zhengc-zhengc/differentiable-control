# sim/tests/test_lat_truck.py
"""LatControllerTruck 测试。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config
from controller.lat_truck import LatControllerTruck
from model.trajectory import generate_straight, generate_circle, TrajectoryAnalyzer

CFG = load_config()


class TestLatTruckBasic:
    def test_on_straight_no_steer(self):
        """车辆在直线上、无偏差 → 转向角应接近 0。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, _, _, _ = ctrl.compute(
            x=50.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True)
        assert abs(steer) < 5.0

    def test_lateral_offset_corrects(self):
        """车辆在直线左偏 2m → 转向角应向右修正（负值，CCW+ 约定负=右转）。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, _, _, _ = ctrl.compute(
            x=50.0, y=2.0, yaw_deg=0.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True)
        assert steer < 0

    def test_circle_has_steer(self):
        """圆弧轨迹上 → 应有前馈转向角。"""
        R = 30.0
        pts = generate_circle(radius=R, speed=5.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, kfar, _, _ = ctrl.compute(
            x=0.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
            yawrate=5.0/R, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True)
        assert abs(steer) > 1.0

    def test_disable_returns_feedback(self):
        """ctrl_enable=False → 输出 = steer_feedback。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, _, _, _ = ctrl.compute(
            x=0.0, y=5.0, yaw_deg=10.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=42.0,
            analyzer=analyzer, ctrl_enable=False)
        assert steer == pytest.approx(42.0)

    def test_is_nn_module(self):
        """控制器应继承 nn.Module。"""
        ctrl = LatControllerTruck(CFG)
        assert isinstance(ctrl, torch.nn.Module)

    def test_has_parameters(self):
        """应有 nn.Parameter：T2-T4, T6 的 y 值（控制设计参数）。
        T5 不参与反馈控制律（仅输出 near_kappa 供监控），归入 buffer。
        """
        ctrl = LatControllerTruck(CFG)
        param_names = {n for n, _ in ctrl.named_parameters()}
        for i in [2, 3, 4, 6]:
            assert f'T{i}_y' in param_names
        # kLh, T1, T5, T7, T8 应为 buffer（物理参数/安全约束/无梯度信号）
        for name in ['kLh', 'T1_y', 'T5_y', 'T7_y', 'T8_y']:
            assert name not in param_names

    def test_has_buffers(self):
        """应有 buffer：8 张表的 x 值 + 内部状态。"""
        ctrl = LatControllerTruck(CFG)
        buf_names = {n for n, _ in ctrl.named_buffers()}
        for i in range(1, 9):
            assert f'T{i}_x' in buf_names
        assert 'steer_fb_prev' in buf_names
        assert 'steer_ff_prev' in buf_names
        assert 'steer_total_prev' in buf_names

    def test_reset_state(self):
        """reset_state 应清零内部状态。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        # 先运行一步让状态非零
        ctrl.compute(x=50.0, y=2.0, yaw_deg=0.0, speed_kph=36.0,
                     yawrate=0.0, steer_feedback=0.0,
                     analyzer=analyzer, ctrl_enable=True)
        assert ctrl.steer_fb_prev.item() != 0.0
        # 重置
        ctrl.reset_state()
        assert ctrl.steer_fb_prev.item() == 0.0
        assert ctrl.steer_ff_prev.item() == 0.0
        assert ctrl.steer_total_prev.item() == 0.0


class TestLatTruckDifferentiable:
    """differentiable=True 模式测试。"""

    def test_output_is_tensor(self):
        """differentiable=True → 输出应为 tensor。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG, differentiable=True)
        steer, kc, kn, kf, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(2.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0), steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        assert isinstance(steer, torch.Tensor)
        assert isinstance(kc, torch.Tensor)
        assert isinstance(kn, torch.Tensor)
        assert isinstance(kf, torch.Tensor)

    def test_gradient_through_T2(self):
        """T2 (prev_time_dist) 梯度应能流过 compute。"""
        ctrl = LatControllerTruck(CFG, differentiable=True)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        steer, _, _, _, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(2.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0), steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        steer.backward()
        assert ctrl.T2_y.grad is not None
        assert ctrl.T2_y.grad.abs().sum() > 0

    def test_no_gradient_through_fixed_params(self):
        """T1, T7, T8, kLh 为 buffer，不应有梯度。"""
        ctrl = LatControllerTruck(CFG, differentiable=True)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        steer, _, _, _, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(2.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.1), steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        steer.backward()
        # buffer 不参与梯度计算
        assert ctrl.T1_y.grad is None
        assert ctrl.T7_y.grad is None
        assert ctrl.T8_y.grad is None
        assert ctrl.kLh.grad is None

    def test_gradient_through_T4(self):
        """T4 (T_dt) 梯度在有航向误差时应流过。"""
        ctrl = LatControllerTruck(CFG, differentiable=True)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        steer, _, _, _, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(0.5),
            yaw_deg=torch.tensor(5.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.1), steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        steer.backward()
        assert ctrl.T4_y.grad is not None
        assert ctrl.T4_y.grad.abs().sum() > 0

    def test_false_matches_v1(self):
        """differentiable=False 输出应与 V1 完全一致。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG, differentiable=False)
        steer, kc, kn, kf, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(2.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0), steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        # V1 参考值
        assert isinstance(steer, (float, int))
        assert steer == pytest.approx(-2.4000000953674316, abs=1e-8)

    def test_false_on_track_v1(self):
        """differentiable=False 在轨迹上 → 转向角=0（V1 一致）。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG, differentiable=False)
        steer, kc, kn, kf, _, _ = ctrl.compute(
            x=50.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True)
        assert steer == pytest.approx(0.0, abs=1e-10)

    def test_differentiable_corrects_direction(self):
        """differentiable=True 修正方向应与 V1 一致。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG, differentiable=True)
        steer, _, _, _, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(2.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0), steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        # 应向右修正（负值）
        assert steer.item() < 0

    def test_disable_differentiable(self):
        """differentiable=True + ctrl_enable=False → 输出 = steer_feedback。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG, differentiable=True)
        steer, _, _, _, _, _ = ctrl.compute(
            x=torch.tensor(0.0), y=torch.tensor(5.0),
            yaw_deg=torch.tensor(10.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0),
            steer_feedback=torch.tensor(42.0),
            analyzer=analyzer, ctrl_enable=False)
        assert isinstance(steer, torch.Tensor)
        assert steer.item() == pytest.approx(42.0)
