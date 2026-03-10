# sim/tests/test_dynamic_vehicle.py
"""动力学车辆适配器测试。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.dynamic_vehicle import DynamicVehicle

# 默认参数（与 plant 项目一致的 B 级车参数）
DEFAULT_PARAMS = {
    'mass': 2440.0,
    'Iz': 9564.8,
    'lf': 1.354,
    'lr': 1.446,
    'wheel_radius': 0.329,
    'drag_coeff': 0.558,
    'frontal_area': 5.903,
    'air_density': 1.225,
    'rolling_coeff': 0.0065,
    'corner_stiff_f': 80000.0,
    'corner_stiff_r': 80000.0,
    'tire_friction_mu': 0.85,
    'track_width': 1.725,
    'steer_ratio': 16.39,
}


class TestDynamicVehicle:
    def test_interface_has_required_attributes(self):
        """适配器必须暴露与 BicycleModel 一致的属性。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=1.0, y=2.0, yaw=0.1, v=5.0, dt=0.02)
        assert isinstance(car.x, torch.Tensor)
        assert isinstance(car.y, torch.Tensor)
        assert isinstance(car.yaw, torch.Tensor)
        assert isinstance(car.v, torch.Tensor)
        assert hasattr(car, 'speed_kph')
        assert hasattr(car, 'yaw_deg')
        assert hasattr(car, 'step')
        assert hasattr(car, 'detach_state')

    def test_straight_line(self):
        """零转角+零加速度，直行应保持 y≈0、yaw≈0。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02)
        for _ in range(500):  # 10s
            car.step(delta=0.0, acc=0.0)
        # 动力学模型有风阻/滚阻会减速，x 应前进但 < 100m
        assert car.x.item() > 50.0
        assert abs(car.y.item()) < 0.5
        assert abs(car.yaw.item()) < 0.01

    def test_acceleration(self):
        """施加正加速度，速度应增大。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=1.0, dt=0.02)
        v0 = car.v.item()
        for _ in range(50):
            car.step(delta=0.0, acc=1.0)
        assert car.v.item() > v0

    def test_steering_causes_lateral_motion(self):
        """施加转向，y 应偏移。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        delta_front = 0.05  # rad，前轮转角
        for _ in range(200):
            car.step(delta=delta_front, acc=0.0)
        assert abs(car.y.item()) > 0.5

    def test_properties(self):
        """speed_kph 和 yaw_deg 属性。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02)
        assert car.speed_kph.item() == pytest.approx(36.0, abs=0.1)
        assert car.yaw_deg.item() == pytest.approx(0.0, abs=0.01)

    def test_detach_state(self):
        """detach_state 应截断梯度链。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                             differentiable=True)
        car.step(delta=0.0, acc=torch.tensor(1.0, requires_grad=True))
        car.detach_state()
        assert not car.x.requires_grad
        assert not car.v.requires_grad

    def test_differentiable_gradient_flows(self):
        """differentiable 模式下梯度应能回传到 acc 输入。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                             differentiable=True)
        acc = torch.tensor(1.0, requires_grad=True)
        car.step(delta=0.0, acc=acc)
        car.v.backward()
        assert acc.grad is not None
        assert acc.grad.item() != 0.0
