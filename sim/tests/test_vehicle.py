# sim/tests/test_vehicle.py
"""运动学自行车模型测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vehicle import BicycleModel


class TestBicycleModel:
    def test_straight_line(self):
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=10.0)
        for _ in range(500):
            car.step(delta=0.0, acc=0.0)
        assert car.x == pytest.approx(100.0, abs=1.0)
        assert car.y == pytest.approx(0.0, abs=0.01)
        assert car.yaw == pytest.approx(0.0, abs=0.01)

    def test_acceleration(self):
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=0.0)
        for _ in range(50):
            car.step(delta=0.0, acc=1.0)
        assert car.v == pytest.approx(1.0, abs=0.05)
        assert car.x > 0

    def test_circular_motion(self):
        L = 3.5
        R = 30.0
        delta = math.atan(L / R)
        speed = 5.0
        car = BicycleModel(wheelbase=L, x=0, y=0, yaw=0, v=speed)
        n_steps = int(2 * math.pi * R / speed / 0.02)
        for _ in range(n_steps):
            car.step(delta=delta, acc=0.0)
        assert car.x == pytest.approx(0.0, abs=2.0)
        assert car.y == pytest.approx(0.0, abs=2.0)

    def test_speed_non_negative(self):
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=1.0)
        for _ in range(200):
            car.step(delta=0.0, acc=-5.0)
        assert car.v >= 0.0
