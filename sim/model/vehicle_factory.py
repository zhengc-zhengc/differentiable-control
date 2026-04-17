# sim/model/vehicle_factory.py
"""车辆模型工厂：根据配置创建车辆模型。"""
import os

from model.vehicle import BicycleModel
from model.dynamic_vehicle import DynamicVehicle
from model.hybrid_dynamic_vehicle import HybridDynamicVehicle
from model.generic_hybrid_vehicle import GenericHybridVehicle
from model.dynamic_vehicle_v2 import VehicleDynamicsV2
from model.truck_trailer_vehicle import TruckTrailerVehicle

# base 动力学模型注册表（hybrid_v2 模式下通过 vehicle.base_model 选择）
_BASE_MODEL_REGISTRY = {
    'dynamic_v2': VehicleDynamicsV2,
}


def resolve_vehicle_geometry(cfg):
    """根据 cfg['vehicle']['model_type'] 解析 (wheelbase, steer_ratio)。

    所有读 plant 几何的地方（sim_loop、lat_truck 等）都应走这里，
    避免新增 plant 类型时遗漏更新。
    """
    veh = cfg['vehicle']
    model_type = veh.get('model_type', 'kinematic')
    if model_type == 'dynamic':
        dyn = cfg['dynamic_vehicle']
        return dyn['lf'] + dyn['lr'], dyn['steer_ratio']
    if model_type == 'hybrid_dynamic':
        hyb = cfg['hybrid_dynamic_vehicle']
        return hyb['lf'] + hyb['lr'], hyb['steer_ratio']
    if model_type == 'hybrid_v2':
        params_key = veh.get('params_section', 'dynamic_v2_vehicle')
        v2p = cfg[params_key]
        return v2p['lf'] + v2p['lr'], v2p['steer_ratio']
    if model_type == 'truck_trailer':
        tt = cfg['truck_trailer_vehicle']
        return tt['L_t'], tt['steering_ratio']
    return veh['wheelbase'], veh['steer_ratio']


def _resolve_checkpoint_path(rel_path):
    """将 checkpoint 相对路径解析为绝对路径（相对于 sim/ 目录）。"""
    if not rel_path or os.path.isabs(rel_path):
        return rel_path
    sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(sim_dir, rel_path))


def create_vehicle(cfg, x=0.0, y=0.0, yaw=0.0, v=0.0,
                   dt=0.02, differentiable=False):
    """根据 cfg['vehicle']['model_type'] 创建车辆模型。

    Args:
        cfg: 完整配置字典
        x, y, yaw, v: 初始状态
        dt: 仿真步长
        differentiable: 是否启用可微模式

    Returns:
        BicycleModel / DynamicVehicle / HybridDynamicVehicle 实例（接口一致）
    """
    model_type = cfg['vehicle'].get('model_type', 'kinematic')

    if model_type == 'kinematic':
        veh = cfg['vehicle']
        return BicycleModel(
            wheelbase=veh['wheelbase'], x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable)

    elif model_type == 'dynamic':
        dyn_params = cfg['dynamic_vehicle']
        return DynamicVehicle(
            params=dyn_params, x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable)

    elif model_type == 'hybrid_dynamic':
        hyb_params = cfg['hybrid_dynamic_vehicle']
        checkpoint = _resolve_checkpoint_path(
            hyb_params.get('checkpoint_path', ''))
        return HybridDynamicVehicle(
            params=hyb_params, x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable,
            checkpoint_path=checkpoint or None)

    elif model_type == 'truck_trailer':
        tt_params = cfg['truck_trailer_vehicle']
        checkpoint = _resolve_checkpoint_path(
            tt_params.get('checkpoint_path', ''))
        return TruckTrailerVehicle(
            params=tt_params, x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable,
            checkpoint_path=checkpoint or None,
            trailer_mass_kg=tt_params.get('default_trailer_mass_kg', None))

    elif model_type == 'hybrid_v2':
        veh_cfg = cfg['vehicle']
        base_model_name = veh_cfg.get('base_model', 'dynamic_v2')
        if base_model_name not in _BASE_MODEL_REGISTRY:
            raise ValueError(
                f"未知 base_model: '{base_model_name}'，"
                f"可用: {list(_BASE_MODEL_REGISTRY.keys())}")
        base_cls = _BASE_MODEL_REGISTRY[base_model_name]
        params_key = veh_cfg.get('params_section', 'dynamic_v2_vehicle')
        params = cfg[params_key]
        checkpoint = _resolve_checkpoint_path(
            veh_cfg.get('checkpoint_path', ''))
        return GenericHybridVehicle(
            params=params, x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable,
            checkpoint_path=checkpoint or None,
            base_model_class=base_cls)

    else:
        raise ValueError(
            f"未知 vehicle.model_type: '{model_type}'，"
            f"支持: 'kinematic', 'dynamic', 'hybrid_dynamic', 'hybrid_v2', "
            f"'truck_trailer'")
