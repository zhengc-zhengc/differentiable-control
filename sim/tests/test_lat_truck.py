# sim/tests/test_lat_truck.py
"""LatControllerTruck 测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config
from controller.lat_truck import LatControllerTruck
from trajectory import generate_straight, generate_circle, TrajectoryAnalyzer

CFG = load_config()


class TestLatTruckBasic:
    def test_on_straight_no_steer(self):
        """车辆在直线上、无偏差 → 转向角应接近 0。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, _ = ctrl.compute(
            x=50.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True)
        assert abs(steer) < 5.0

    def test_lateral_offset_corrects(self):
        """车辆在直线左偏 2m → 转向角应向右修正（负值，CCW+ 约定负=右转）。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, _ = ctrl.compute(
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
        steer, _, _, kfar = ctrl.compute(
            x=0.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
            yawrate=5.0/R, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True)
        assert abs(steer) > 1.0

    def test_disable_returns_feedback(self):
        """ctrl_enable=False → 输出 = steer_feedback。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, _ = ctrl.compute(
            x=0.0, y=5.0, yaw_deg=10.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=42.0,
            analyzer=analyzer, ctrl_enable=False)
        assert steer == pytest.approx(42.0)
