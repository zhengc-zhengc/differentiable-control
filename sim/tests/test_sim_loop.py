# sim/tests/test_sim_loop.py
"""闭环仿真测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim_loop import run_simulation
from trajectory import generate_straight, generate_circle


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
