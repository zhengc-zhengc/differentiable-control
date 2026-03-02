# sim/tests/test_common.py
"""common.py 基础组件测试。V2: torch 化 + 梯度流测试。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common import (lookup1d, rate_limit, normalize_angle, clamp, sign,
                     PID, IIR, TrajectoryPoint,
                     smooth_clamp, smooth_sign, smooth_step)


class TestLookup1d:
    def test_exact_match(self):
        tx = torch.tensor([0.0, 10.0, 20.0])
        ty = torch.tensor([0.0, 1.0, 2.0])
        assert lookup1d(tx, ty, torch.tensor(10.0)).item() == pytest.approx(1.0)

    def test_interpolation(self):
        tx = torch.tensor([0.0, 10.0, 20.0])
        ty = torch.tensor([0.0, 1.0, 2.0])
        assert lookup1d(tx, ty, torch.tensor(5.0)).item() == pytest.approx(0.5)

    def test_clamp_low(self):
        tx = torch.tensor([0.0, 10.0])
        ty = torch.tensor([0.0, 1.0])
        assert lookup1d(tx, ty, torch.tensor(-5.0)).item() == pytest.approx(0.0)

    def test_clamp_high(self):
        tx = torch.tensor([0.0, 10.0])
        ty = torch.tensor([0.0, 1.0])
        assert lookup1d(tx, ty, torch.tensor(15.0)).item() == pytest.approx(1.0)

    def test_single_point(self):
        tx = torch.tensor([5.0])
        ty = torch.tensor([3.0])
        assert lookup1d(tx, ty, torch.tensor(0.0)).item() == pytest.approx(3.0)
        assert lookup1d(tx, ty, torch.tensor(10.0)).item() == pytest.approx(3.0)

    def test_gradient_flows_through_y(self):
        """验证 lookup1d 对 table_y 的梯度流。"""
        tx = torch.tensor([0.0, 10.0, 20.0])
        ty = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        result = lookup1d(tx, ty, torch.tensor(5.0))
        result.backward()
        assert ty.grad is not None
        # x=5 在 [0,10] 区间，t=0.5，所以 y = y0*(1-t) + y1*t
        # dy/dy0 = 0.5, dy/dy1 = 0.5, dy/dy2 = 0
        assert ty.grad[0].item() == pytest.approx(0.5)
        assert ty.grad[1].item() == pytest.approx(0.5)
        assert ty.grad[2].item() == pytest.approx(0.0)

    def test_accepts_plain_float(self):
        """lookup1d 也接受非 tensor 的 x 输入。"""
        tx = torch.tensor([0.0, 10.0])
        ty = torch.tensor([0.0, 1.0])
        result = lookup1d(tx, ty, 5.0)
        assert result.item() == pytest.approx(0.5)


class TestRateLimit:
    def test_within_limit(self):
        result = rate_limit(0.0, 5.0, 300, 0.02)
        assert result.item() == pytest.approx(5.0)

    def test_rate_clamped_up(self):
        result = rate_limit(0.0, 10.0, 120, 0.02)
        assert result.item() == pytest.approx(2.4)

    def test_rate_clamped_down(self):
        result = rate_limit(10.0, 0.0, 120, 0.02)
        assert result.item() == pytest.approx(10 - 2.4)


class TestNormalizeAngle:
    def test_zero(self):
        assert normalize_angle(0).item() == pytest.approx(0, abs=1e-6)

    def test_pi(self):
        assert abs(normalize_angle(math.pi).item()) == pytest.approx(math.pi, abs=1e-6)

    def test_wrap_positive(self):
        # atan2 版 normalize 在 3*pi 处返回 -pi 或 +pi，abs 应 ≈ pi
        assert abs(normalize_angle(3 * math.pi).item()) == pytest.approx(math.pi, abs=1e-6)

    def test_wrap_negative(self):
        assert abs(normalize_angle(-3 * math.pi).item()) == pytest.approx(math.pi, abs=1e-6)

    def test_just_over_pi(self):
        assert normalize_angle(math.pi + 0.1).item() == pytest.approx(-math.pi + 0.1, abs=1e-6)

    def test_returns_tensor(self):
        result = normalize_angle(1.5)
        assert isinstance(result, torch.Tensor)


class TestClamp:
    def test_basic(self):
        assert clamp(torch.tensor(5.0), 0, 10).item() == pytest.approx(5.0)
        assert clamp(torch.tensor(-1.0), 0, 10).item() == pytest.approx(0.0)
        assert clamp(torch.tensor(15.0), 0, 10).item() == pytest.approx(10.0)

    def test_differentiable_mode(self):
        result = clamp(torch.tensor(5.0), 0, 10, differentiable=True)
        assert isinstance(result, torch.Tensor)
        # smooth_clamp 结果应接近真实值
        assert result.item() == pytest.approx(5.0, abs=0.5)


class TestSign:
    def test_positive(self):
        assert sign(5.0).item() == pytest.approx(1.0)

    def test_negative(self):
        assert sign(-5.0).item() == pytest.approx(-1.0)

    def test_zero(self):
        assert sign(0.0).item() == pytest.approx(0.0)

    def test_differentiable_mode(self):
        result = sign(torch.tensor(5.0), differentiable=True)
        assert result.item() == pytest.approx(1.0, abs=0.1)


class TestSmoothFunctions:
    def test_smooth_clamp_gradient_at_boundary(self):
        """smooth_clamp 边界附近梯度非零（相比 hard clamp 梯度为零）。"""
        # 使用较大 temp 确保边界附近有可观梯度
        x = torch.tensor(10.0, requires_grad=True)
        result = smooth_clamp(x, 0.0, 10.0, temp=0.5)
        result.backward()
        assert x.grad is not None
        assert abs(x.grad.item()) > 1e-2  # 非零梯度

    def test_smooth_sign_gradient_at_zero(self):
        """smooth_sign 零点处梯度非零。"""
        x = torch.tensor(0.0, requires_grad=True)
        result = smooth_sign(x, temp=0.01)
        result.backward()
        assert x.grad is not None
        assert abs(x.grad.item()) > 1.0  # 1/temp = 100，梯度很大

    def test_smooth_step_basic(self):
        assert smooth_step(torch.tensor(10.0), 5.0).item() > 0.9
        assert smooth_step(torch.tensor(0.0), 5.0).item() < 0.1

    def test_smooth_clamp_within_range(self):
        """范围内值应接近输入。"""
        result = smooth_clamp(torch.tensor(5.0), 0.0, 10.0, temp=0.1)
        assert result.item() == pytest.approx(5.0, abs=0.5)

    def test_smooth_clamp_outside_range(self):
        """范围外值应接近边界。"""
        result = smooth_clamp(torch.tensor(20.0), 0.0, 10.0, temp=0.1)
        assert result.item() == pytest.approx(10.0, abs=0.5)


class TestPID:
    def test_proportional_only(self):
        pid = PID()
        result = pid.control(2.0, 0.02, kp=1.0, ki=0.0, kd=0.0,
                             integrator_enable=False, sat=1.0)
        assert result.item() == pytest.approx(2.0)

    def test_integral_accumulation(self):
        pid = PID()
        pid.control(1.0, 0.02, kp=0.0, ki=1.0, kd=0.0,
                    integrator_enable=True, sat=10.0)  # integral = 0.02
        pid.control(1.0, 0.02, kp=0.0, ki=1.0, kd=0.0,
                    integrator_enable=True, sat=10.0)  # integral = 0.04
        result = pid.control(1.0, 0.02, kp=0.0, ki=1.0, kd=0.0,
                             integrator_enable=True, sat=10.0)  # integral = 0.06
        assert result.item() == pytest.approx(0.06)

    def test_integral_saturation(self):
        pid = PID()
        for _ in range(100):
            pid.control(1.0, 0.02, kp=0.0, ki=1.0, kd=0.0,
                        integrator_enable=True, sat=0.01)
        result = pid.control(1.0, 0.02, kp=0.0, ki=1.0, kd=0.0,
                             integrator_enable=True, sat=0.01)
        assert result.item() == pytest.approx(0.01)

    def test_reset(self):
        pid = PID()
        pid.control(1.0, 0.02, kp=1.0, ki=1.0, kd=0.0,
                    integrator_enable=True, sat=10.0)
        pid.reset()
        assert pid.integral.item() == 0.0
        assert pid.prev_error.item() == 0.0


class TestIIR:
    def test_basic(self):
        iir = IIR(alpha=0.5)
        y1 = iir.update(1.0)
        assert isinstance(y1, torch.Tensor)
        assert y1.item() == pytest.approx(1.0)  # x - 0.5*0 = 1.0
        y2 = iir.update(1.0)
        assert y2.item() == pytest.approx(0.5)  # x - 0.5*1.0 = 0.5

    def test_reset(self):
        iir = IIR(alpha=0.5)
        iir.update(1.0)
        iir.reset()
        assert iir.y_prev.item() == 0.0


class TestTrajectoryPoint:
    def test_fields(self):
        pt = TrajectoryPoint(x=1.0, y=2.0, theta=0.1, kappa=0.01,
                             v=5.0, a=0.0, s=10.0, t=0.0)
        assert pt.x == 1.0
        assert pt.v == 5.0
