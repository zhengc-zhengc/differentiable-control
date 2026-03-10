# sim/tests/test_trajectory.py
"""轨迹生成与分析器测试。V2: torch 化。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.trajectory import (generate_straight, generate_circle, generate_sine,
                              generate_combined, generate_lane_change,
                              generate_double_lane_change, generate_s_curve,
                              TrajectoryAnalyzer)


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


class TestLaneChange:
    def test_returns_points(self):
        pts = generate_lane_change(lane_width=3.5, change_length=50.0, speed=5.0)
        assert len(pts) > 0
        assert pts[0].v == pytest.approx(5.0)

    def test_end_position(self):
        """换道终点 y 应接近 lane_width。"""
        pts = generate_lane_change(lane_width=3.5, change_length=50.0, speed=5.0)
        assert pts[-1].y == pytest.approx(3.5, abs=0.1)

    def test_high_speed(self):
        """高速换道（60 km/h）应正常生成。"""
        pts = generate_lane_change(lane_width=3.5, change_length=100.0,
                                   speed=60.0 / 3.6)
        assert len(pts) > 0
        assert pts[0].v == pytest.approx(60.0 / 3.6, abs=0.01)


class TestDoubleLaneChange:
    def test_returns_points(self):
        pts = generate_double_lane_change(lane_width=3.5, change_length=50.0,
                                          speed=5.0)
        assert len(pts) > 0
        assert pts[0].v == pytest.approx(5.0)

    def test_returns_to_original_lane(self):
        """双换道终点 y 应接近 0（返回原车道）。"""
        pts = generate_double_lane_change(lane_width=3.5, change_length=50.0,
                                          speed=5.0)
        assert pts[-1].y == pytest.approx(0.0, abs=0.1)

    def test_curvature_sign_changes(self):
        """双换道应包含正负曲率。"""
        pts = generate_double_lane_change(lane_width=3.5, change_length=50.0,
                                          speed=5.0)
        kappas = [p.kappa for p in pts]
        assert max(kappas) > 0.001, "应有正曲率"
        assert min(kappas) < -0.001, "应有负曲率"


class TestSCurve:
    def test_returns_points(self):
        pts = generate_s_curve(radius=50.0, arc_angle=math.pi / 4, speed=5.0)
        assert len(pts) > 0
        assert pts[0].v == pytest.approx(5.0)

    def test_exit_heading_near_zero(self):
        """对称 S 弯出口航向应接近 0。"""
        pts = generate_s_curve(radius=50.0, arc_angle=math.pi / 4, speed=5.0)
        assert pts[-1].theta == pytest.approx(0.0, abs=0.05)

    def test_curvature_sign_changes(self):
        """S 弯应包含正负曲率（左转+右转）。"""
        pts = generate_s_curve(radius=50.0, arc_angle=math.pi / 4, speed=5.0)
        kappas = [p.kappa for p in pts]
        assert max(kappas) > 0.001, "应有正曲率（左转段）"
        assert min(kappas) < -0.001, "应有负曲率（右转段）"


class TestMultiSpeedTrajectories:
    """验证多速度轨迹生成器正确嵌入目标速度。"""

    @pytest.mark.parametrize("speed_mps", [25/3.6, 40/3.6, 50/3.6, 60/3.6])
    def test_lane_change_speed_embedded(self, speed_mps):
        """不同速度的换道轨迹应正确嵌入对应速度。"""
        pts = generate_lane_change(lane_width=3.5, change_length=80.0,
                                   speed=speed_mps)
        assert pts[0].v == pytest.approx(speed_mps, abs=0.01)
        assert len(pts) > 50


class TestAnalyzer:
    def test_nearest_by_position(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_position(50.0, 0.0)
        assert pt.x == pytest.approx(50.0, abs=0.5)

    def test_nearest_by_position_tensor_input(self):
        """query_nearest_by_position 接受 tensor 输入。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_position(
            torch.tensor(50.0), torch.tensor(0.0))
        assert pt.x == pytest.approx(50.0, abs=0.5)

    def test_nearest_by_relative_time(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_relative_time(1.0)
        assert pt.x == pytest.approx(10.0, abs=0.5)
        assert pt.v == pytest.approx(10.0)

    def test_nearest_by_relative_time_tensor(self):
        """query_nearest_by_relative_time 接受 tensor 输入。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_relative_time(torch.tensor(1.0))
        assert pt.x == pytest.approx(10.0, abs=0.5)

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
        assert d.item() == pytest.approx(0.0, abs=0.1)
        assert s_dot.item() == pytest.approx(10.0, abs=0.5)

    def test_frenet_lateral_offset(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        matched = analyzer.query_nearest_by_position(50.0, 2.0)
        s, s_dot, d, d_dot = analyzer.to_frenet(50.0, 2.0, 0.0, 10.0, matched)
        assert d.item() == pytest.approx(2.0, abs=0.2)

    def test_frenet_returns_tensors(self):
        """to_frenet 返回值为 torch.Tensor。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        matched = analyzer.query_nearest_by_position(50.0, 0.0)
        s, s_dot, d, d_dot = analyzer.to_frenet(50.0, 0.0, 0.0, 10.0, matched)
        assert isinstance(s, torch.Tensor)
        assert isinstance(s_dot, torch.Tensor)
        assert isinstance(d, torch.Tensor)
        assert isinstance(d_dot, torch.Tensor)

    def test_has_precomputed_tensors(self):
        """TrajectoryAnalyzer 应预计算 _xs, _ys 张量。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        assert isinstance(analyzer._xs, torch.Tensor)
        assert isinstance(analyzer._ys, torch.Tensor)
        assert len(analyzer._xs) == len(pts)
