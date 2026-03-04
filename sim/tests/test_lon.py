# sim/tests/test_lon.py
"""LonController 简化版测试。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config
from controller.lon import LonController
from model.trajectory import generate_straight, TrajectoryAnalyzer

CFG = load_config()


class TestLonBasic:
    def test_on_track_no_correction(self):
        """车辆在轨迹上、速度匹配 → 加速度应接近 0。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG)
        acc = ctrl.compute(
            x=10.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        assert abs(acc) < 0.5

    def test_too_slow_accelerates(self):
        """车速比参考慢 → 应输出正加速度。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG)
        acc = ctrl.compute(
            x=10.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        assert acc > 0

    def test_too_fast_decelerates(self):
        """车速比参考快 → 应输出负加速度。"""
        pts = generate_straight(length=200, speed=5.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG)
        acc = ctrl.compute(
            x=10.0, y=0.0, yaw_deg=0.0, speed_kph=54.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        assert acc < 0

    def test_acc_within_limits(self):
        """加速度应在 L1/L2 限幅范围内。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG)
        for _ in range(100):
            acc = ctrl.compute(
                x=10.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
                accel_mps2=0.0, curvature_far=0.0,
                analyzer=analyzer, t_now=1.0,
                ctrl_enable=True, ctrl_first_active=False)
            assert -4.0 <= acc <= 2.0

    def test_is_nn_module(self):
        """控制器应继承 nn.Module。"""
        ctrl = LonController(CFG)
        assert isinstance(ctrl, torch.nn.Module)

    def test_has_parameters(self):
        """应有 nn.Parameter：PID 增益 + 切换速度。L1-L5 为 buffer。"""
        ctrl = LonController(CFG)
        param_names = {n for n, _ in ctrl.named_parameters()}
        for name in ['station_kp', 'station_ki', 'low_speed_kp', 'low_speed_ki',
                     'high_speed_kp', 'high_speed_ki', 'switch_speed']:
            assert name in param_names
        # L1-L5 应为 buffer（物理限制/安全约束）
        for i in range(1, 6):
            assert f'L{i}_y' not in param_names

    def test_has_buffers(self):
        """应有 buffer：5 张表的 x 值 + acc_out_prev。"""
        ctrl = LonController(CFG)
        buf_names = {n for n, _ in ctrl.named_buffers()}
        for i in range(1, 6):
            assert f'L{i}_x' in buf_names
        assert 'acc_out_prev' in buf_names

    def test_reset_state(self):
        """reset_state 应清零内部状态。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG)
        # 先运行几步让状态非零
        for i in range(5):
            ctrl.compute(x=10.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
                         accel_mps2=0.0, curvature_far=0.0,
                         analyzer=analyzer, t_now=1.0,
                         ctrl_enable=True, ctrl_first_active=(i == 0))
        assert ctrl.acc_out_prev.item() != 0.0
        # 重置
        ctrl.reset_state()
        assert ctrl.acc_out_prev.item() == 0.0
        assert ctrl.station_error_fnl_prev == 0.0


class TestLonDifferentiable:
    """differentiable=True 模式测试。"""

    def test_output_is_tensor(self):
        """differentiable=True → 输出应为 tensor。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG, differentiable=True)
        acc = ctrl.compute(
            x=torch.tensor(10.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(18.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=True)
        assert isinstance(acc, torch.Tensor)

    def test_gradient_through_kp(self):
        """low_speed_kp 梯度应能流过 compute。"""
        ctrl = LonController(CFG, differentiable=True)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        acc = ctrl.compute(
            x=torch.tensor(10.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(18.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=True)
        acc.backward()
        assert ctrl.low_speed_kp.grad is not None
        assert abs(ctrl.low_speed_kp.grad.item()) > 0

    def test_gradient_through_high_speed_kp(self):
        """high_speed_kp 梯度在高速时应流过。"""
        ctrl = LonController(CFG, differentiable=True)
        pts = generate_straight(length=200, speed=5.0)
        analyzer = TrajectoryAnalyzer(pts)
        # 高速：54 kph = 15 m/s > switch_speed=3.0
        acc = ctrl.compute(
            x=torch.tensor(10.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(54.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=True)
        acc.backward()
        assert ctrl.high_speed_kp.grad is not None
        assert abs(ctrl.high_speed_kp.grad.item()) > 0

    def test_gradient_through_switch_speed(self):
        """switch_speed 梯度应流过（通过 smooth_step 混合）。"""
        ctrl = LonController(CFG, differentiable=True)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        acc = ctrl.compute(
            x=torch.tensor(10.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(18.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=True)
        acc.backward()
        assert ctrl.switch_speed.grad is not None

    def test_no_gradient_through_L_tables(self):
        """L1-L5 为 buffer，不应有梯度。"""
        ctrl = LonController(CFG, differentiable=True)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        for i in range(10):
            acc = ctrl.compute(
                x=torch.tensor(10.0), y=torch.tensor(0.0),
                yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(5.0),
                accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
                analyzer=analyzer, t_now=1.0,
                ctrl_enable=True, ctrl_first_active=(i == 0))
        acc.backward()
        for i in range(1, 6):
            assert getattr(ctrl, f'L{i}_y').grad is None

    def test_false_matches_v1(self):
        """differentiable=False 输出应与 V1 完全一致。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG, differentiable=False)
        acc = ctrl.compute(
            x=10.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        # V1 参考值
        assert acc == pytest.approx(0.04830000177025795, abs=1e-10)

    def test_false_on_track_v1(self):
        """differentiable=False 在轨迹上 → 加速度接近 0（V1 一致）。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG, differentiable=False)
        acc = ctrl.compute(
            x=10.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        assert abs(acc) < 1e-10

    def test_differentiable_direction(self):
        """differentiable=True 加速/减速方向应与 V1 一致。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        # 太慢 -> 应加速
        ctrl1 = LonController(CFG, differentiable=True)
        acc1 = ctrl1.compute(
            x=torch.tensor(10.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(18.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=True)
        assert acc1.item() > 0
        # 太快 -> 应减速
        pts2 = generate_straight(length=200, speed=5.0)
        analyzer2 = TrajectoryAnalyzer(pts2)
        ctrl2 = LonController(CFG, differentiable=True)
        acc2 = ctrl2.compute(
            x=torch.tensor(10.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(54.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer2, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=True)
        assert acc2.item() < 0

    def test_station_kp_gradient(self):
        """station_kp 梯度在有站位误差时应流过。"""
        ctrl = LonController(CFG, differentiable=True)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        # x=50 vs t_now=0 -> 站位误差较大
        acc = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer, t_now=0.0,
            ctrl_enable=True, ctrl_first_active=True)
        acc.backward()
        assert ctrl.station_kp.grad is not None
        assert abs(ctrl.station_kp.grad.item()) > 0
