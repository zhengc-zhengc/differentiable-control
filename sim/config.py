# sim/config.py
"""配置文件加载器。"""
import os
import yaml


def load_config(path: str | None = None) -> dict:
    """加载 YAML 配置文件。默认加载 configs/default.yaml。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def table_from_config(entries: list[list[float]]) -> list[tuple[float, float]]:
    """将 YAML 格式的表 [[idx, val], ...] 转为 lookup1d 需要的 tuple 列表。"""
    return [(row[0], row[1]) for row in entries]
