# sim/config.py
"""配置文件加载器。V2: torch 表加载 + 调参结果保存。"""
import os
import subprocess
from datetime import datetime

import torch
import yaml


def load_config(path: str | None = None) -> dict:
    """加载 YAML 配置文件。默认加载 configs/default.yaml。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def apply_plant_override(cfg: dict, plant: str) -> None:
    """将 --plant 参数应用到配置。hybrid_v2 需要额外默认值。"""
    cfg['vehicle']['model_type'] = plant
    if plant == 'hybrid_v2':
        cfg['vehicle'].setdefault('base_model', 'dynamic_v2')
        cfg['vehicle'].setdefault('params_section', 'dynamic_v2_vehicle')
        cfg['vehicle'].setdefault(
            'checkpoint_path',
            'configs/checkpoints/best_error_model_v2.pth')


def table_from_config(entries: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """将 YAML 格式的表 [[idx, val], ...] 转为 (x_tensor, y_tensor)。
    返回的 y_tensor 可作为 nn.Parameter 进行梯度优化。
    """
    xs = torch.tensor([row[0] for row in entries], dtype=torch.float32)
    ys = torch.tensor([row[1] for row in entries], dtype=torch.float32)
    return xs, ys


def _get_commit_hash() -> str:
    """获取当前 git commit 短哈希。"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return 'unknown'


def _tensor_to_python(obj):
    """递归将 torch.Tensor 转为 Python float/list，用于 YAML 序列化。"""
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return float(obj.item())
        return [float(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _tensor_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tensor_to_python(v) for v in obj]
    return obj


def save_tuned_config(cfg: dict, output_dir: str | None = None,
                      meta: dict | None = None) -> str:
    """保存调参后的配置到 YAML 文件。

    Args:
        cfg: 配置字典（结构与 default.yaml 一致）
        output_dir: 输出目录，默认 sim/configs/tuned/
        meta: 元信息（final_loss, epochs, trajectories 等）

    Returns:
        保存的文件路径
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'configs', 'tuned')
    os.makedirs(output_dir, exist_ok=True)

    commit_hash = _get_commit_hash()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'tuned_{commit_hash}_{timestamp}.yaml'
    filepath = os.path.join(output_dir, filename)

    # 递归转换 tensor → Python 类型
    cfg_serializable = _tensor_to_python(cfg)

    # 附加 meta 信息
    if meta is not None:
        cfg_serializable['_meta'] = {
            'commit': commit_hash,
            'timestamp': timestamp,
            **_tensor_to_python(meta),
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(cfg_serializable, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    return filepath
