# sim/tests/test_trajectory.py
"""轨迹生成与分析器测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trajectory import (generate_straight, generate_circle, generate_sine,
                         generate_combined, TrajectoryAnalyzer)


class TestStraightTrajectory:
    def test_length_and_speed(self):
        pts = generate_straight(length=100, speed=10.0, dt=0.02)
        assert len(pts) > 0
        assert pts[0].v == pytest.approx(10.0)
        assert len(pts) == pytest.approx(500, abs=5)

    def test_heading_zero(self):
        pts = generate_straight(length=50, speed=5.0)
        for p in pts:
            assert p.theta == pytest.approx(0.0, abs=1e-6)
            assert p.kappa == pytest.approx(0.0, abs=1e-6)


class TestCircleTrajectory:
    def test_constant_curvature(self):
        R = 30.0
        pts = generate_circle(radius=R, speed=5.0, arc_angle=math.pi)
        for p in pts:
            assert p.kappa == pytest.approx(1.0 / R, abs=1e-6)

    def test_arc_length(self):
        R = 30.0
        pts = generate_circle(radius=R, speed=5.0, arc_angle=math.pi)
        total_s = pts[-1].s
        expected = R * math.pi
        assert total_s == pytest.approx(expected, rel=0.02)


class TestSineTrajectory:
    def test_returns_points(self):
        pts = generate_sine(amplitude=5.0, wavelength=50.0,
                            n_waves=2, speed=5.0)
        assert len(pts) > 0
        assert pts[0].x == pytest.approx(0, abs=0.1)
        assert pts[0].y == pytest.approx(0, abs=0.1)


class TestAnalyzer:
    def test_nearest_by_position(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_position(50.0, 0.0)
        assert pt.x == pytest.approx(50.0, abs=0.5)

    def test_nearest_by_relative_time(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_relative_time(1.0)
        assert pt.x == pytest.approx(10.0, abs=0.5)
        assert pt.v == pytest.approx(10.0)

    def test_time_clamp_at_end(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_relative_time(999.0)
        assert pt.x == pytest.approx(pts[-1].x, abs=0.5)

    def test_frenet_on_track(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        matched = analyzer.query_nearest_by_position(50.0, 0.0)
        s, s_dot, d, d_dot = analyzer.to_frenet(50.0, 0.0, 0.0, 10.0, matched)
        assert d == pytest.approx(0.0, abs=0.1)
        assert s_dot == pytest.approx(10.0, abs=0.5)

    def test_frenet_lateral_offset(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        matched = analyzer.query_nearest_by_position(50.0, 2.0)
        s, s_dot, d, d_dot = analyzer.to_frenet(50.0, 2.0, 0.0, 10.0, matched)
        assert d == pytest.approx(2.0, abs=0.2)
