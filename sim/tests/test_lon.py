# sim/tests/test_lon.py
"""LonController 简化版测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config
from controller.lon import LonController
from trajectory import generate_straight, TrajectoryAnalyzer

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
