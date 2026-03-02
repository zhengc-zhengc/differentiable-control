# sim/tests/test_vehicle.py
"""运动学自行车模型测试。V2: torch 化。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.vehicle import BicycleModel


class TestBicycleModel:
    def test_straight_line(self):
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=10.0)
        for _ in range(500):
            car.step(delta=torch.tensor(0.0), acc=torch.tensor(0.0))
        assert car.x.item() == pytest.approx(100.0, abs=1.0)
        assert car.y.item() == pytest.approx(0.0, abs=0.01)
        assert car.yaw.item() == pytest.approx(0.0, abs=0.01)

    def test_acceleration(self):
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=0.0)
        for _ in range(50):
            car.step(delta=torch.tensor(0.0), acc=torch.tensor(1.0))
        assert car.v.item() == pytest.approx(1.0, abs=0.05)
        assert car.x.item() > 0

    def test_circular_motion(self):
        L = 3.5
        R = 30.0
        delta = math.atan(L / R)
        speed = 5.0
        car = BicycleModel(wheelbase=L, x=0, y=0, yaw=0, v=speed)
        n_steps = int(2 * math.pi * R / speed / 0.02)
        for _ in range(n_steps):
            car.step(delta=torch.tensor(delta), acc=torch.tensor(0.0))
        assert car.x.item() == pytest.approx(0.0, abs=2.0)
        assert car.y.item() == pytest.approx(0.0, abs=2.0)

    def test_speed_non_negative(self):
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=1.0)
        for _ in range(200):
            car.step(delta=torch.tensor(0.0), acc=torch.tensor(-5.0))
        assert car.v.item() >= 0.0

    def test_state_is_tensor(self):
        car = BicycleModel(wheelbase=3.5, x=1.0, y=2.0, yaw=0.1, v=5.0)
        assert isinstance(car.x, torch.Tensor)
        assert isinstance(car.y, torch.Tensor)
        assert isinstance(car.yaw, torch.Tensor)
        assert isinstance(car.v, torch.Tensor)

    def test_properties(self):
        car = BicycleModel(wheelbase=3.5, v=10.0)
        assert car.speed_kph.item() == pytest.approx(36.0)
        assert car.yaw_deg.item() == pytest.approx(0.0)

    def test_step_accepts_float(self):
        """step 应也接受 float 输入。"""
        car = BicycleModel(wheelbase=3.5, v=5.0)
        car.step(delta=0.0, acc=0.0)
        assert car.x.item() == pytest.approx(5.0 * 0.02, abs=0.001)

    def test_differentiable_mode_softplus(self):
        """differentiable 模式使用 softplus 而非 clamp。"""
        car = BicycleModel(wheelbase=3.5, v=0.1, differentiable=True)
        car.step(delta=0.0, acc=-10.0)
        # softplus 保证正值，但不会精确为 0
        assert car.v.item() > 0.0
