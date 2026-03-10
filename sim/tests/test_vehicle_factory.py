# sim/tests/test_vehicle_factory.py
"""车辆工厂测试。"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.vehicle import BicycleModel
from model.dynamic_vehicle import DynamicVehicle
from model.vehicle_factory import create_vehicle
from config import load_config


class TestVehicleFactory:
    def test_default_creates_kinematic(self):
        """默认配置应创建 BicycleModel。"""
        cfg = load_config()
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        assert isinstance(car, BicycleModel)

    def test_kinematic_explicit(self):
        """显式指定 kinematic。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'kinematic'
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        assert isinstance(car, BicycleModel)

    def test_dynamic_explicit(self):
        """显式指定 dynamic。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'dynamic'
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        assert isinstance(car, DynamicVehicle)

    def test_dynamic_has_correct_interface(self):
        """dynamic 模型暴露与 kinematic 一致的接口。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'dynamic'
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        car.step(delta=0.0, acc=0.0)
        assert hasattr(car, 'x')
        assert hasattr(car, 'speed_kph')
        assert hasattr(car, 'yaw_deg')
        assert hasattr(car, 'detach_state')

    def test_invalid_model_type_raises(self):
        """无效 model_type 应报错。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'invalid'
        with pytest.raises(ValueError, match='model_type'):
            create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)

    def test_differentiable_forwarded(self):
        """differentiable 参数应传递到模型。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'dynamic'
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                             differentiable=True)
        assert car.differentiable is True
