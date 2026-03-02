# sim/tests/test_common.py
"""common.py 基础组件测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common import lookup1d, rate_limit, normalize_angle, PID, TrajectoryPoint


class TestLookup1d:
    def test_exact_match(self):
        table = [(0, 0), (10, 1), (20, 2)]
        assert lookup1d(table, 10) == 1.0

    def test_interpolation(self):
        table = [(0, 0), (10, 1), (20, 2)]
        assert lookup1d(table, 5) == pytest.approx(0.5)

    def test_clamp_low(self):
        table = [(0, 0), (10, 1)]
        assert lookup1d(table, -5) == 0.0

    def test_clamp_high(self):
        table = [(0, 0), (10, 1)]
        assert lookup1d(table, 15) == 1.0

    def test_single_point(self):
        table = [(5, 3)]
        assert lookup1d(table, 0) == 3.0
        assert lookup1d(table, 10) == 3.0


class TestRateLimit:
    def test_within_limit(self):
        result = rate_limit(0, 5, 300, 0.02)
        assert result == pytest.approx(5.0)

    def test_rate_clamped_up(self):
        result = rate_limit(0, 10, 120, 0.02)
        assert result == pytest.approx(2.4)

    def test_rate_clamped_down(self):
        result = rate_limit(10, 0, 120, 0.02)
        assert result == pytest.approx(10 - 2.4)


class TestNormalizeAngle:
    def test_zero(self):
        assert normalize_angle(0) == pytest.approx(0)

    def test_pi(self):
        assert abs(normalize_angle(math.pi)) == pytest.approx(math.pi)

    def test_wrap_positive(self):
        assert normalize_angle(3 * math.pi) == pytest.approx(math.pi, abs=1e-10)

    def test_wrap_negative(self):
        assert normalize_angle(-3 * math.pi) == pytest.approx(math.pi, abs=1e-10)

    def test_just_over_pi(self):
        assert normalize_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)


class TestPID:
    def test_proportional_only(self):
        pid = PID(kp=1.0, ki=0.0, kd=0.0, integrator_enable=False,
                  integrator_saturation=1.0)
        assert pid.control(2.0, 0.02) == pytest.approx(2.0)

    def test_integral_accumulation(self):
        pid = PID(kp=0.0, ki=1.0, kd=0.0, integrator_enable=True,
                  integrator_saturation=10.0)
        pid.control(1.0, 0.02)  # integral = 0.02
        pid.control(1.0, 0.02)  # integral = 0.04
        assert pid.control(1.0, 0.02) == pytest.approx(0.06)

    def test_integral_saturation(self):
        pid = PID(kp=0.0, ki=1.0, kd=0.0, integrator_enable=True,
                  integrator_saturation=0.01)
        for _ in range(100):
            pid.control(1.0, 0.02)
        result = pid.control(1.0, 0.02)
        assert result == pytest.approx(0.01)

    def test_reset(self):
        pid = PID(kp=1.0, ki=1.0, kd=0.0, integrator_enable=True,
                  integrator_saturation=10.0)
        pid.control(1.0, 0.02)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0


class TestTrajectoryPoint:
    def test_fields(self):
        pt = TrajectoryPoint(x=1.0, y=2.0, theta=0.1, kappa=0.01,
                             v=5.0, a=0.0, s=10.0, t=0.0)
        assert pt.x == 1.0
        assert pt.v == 5.0
