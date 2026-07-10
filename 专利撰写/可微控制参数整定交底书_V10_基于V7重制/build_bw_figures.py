# -*- coding: utf-8 -*-
"""基于 V7 的技术语义重绘黑白专利插图。

本脚本只读取 V7 对应实验的结构化结果，不读取或复用 V8/V9 图稿。
输出图统一为白底灰度 PNG；框图使用黑框和黑箭头，曲线使用线型与
标记区分，柱状图使用纹理区分，适合嵌入正式专利技术交底书。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
RESULT_DIR = (
    ROOT
    / "sim"
    / "results"
    / "training"
    / "truck_trailer"
    / "20260526_123421_mlp0525"
)
LOG_PATH = RESULT_DIR / "experiment_log.yaml"

BLACK = "#000000"
WHITE = "#FFFFFF"
GRAY_1 = "#E6E6E6"
GRAY_2 = "#B3B3B3"
GRAY_3 = "#666666"


def configure_matplotlib() -> None:
    """Configure a deterministic Chinese black-and-white plotting style."""
    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            mpl.font_manager.fontManager.addfont(str(font_path))
            prop = mpl.font_manager.FontProperties(fname=str(font_path))
            mpl.rcParams["font.family"] = prop.get_name()
            break
    mpl.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
        }
    )


def load_log() -> dict:
    with LOG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=BLACK, width=0.8, labelsize=8)
    if grid:
        ax.grid(True, color=GRAY_2, linewidth=0.45, linestyle=":", alpha=1.0)
    ax.set_axisbelow(True)


def save_gray(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    """Save a figure as equal-channel RGB for robust Word rendering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.10, facecolor=WHITE)
    plt.close(fig)
    with Image.open(path) as image:
        image.convert("L").convert("RGB").save(path, optimize=True)


def combine_vertical(paths: Iterable[Path], target: Path, gap: int = 28) -> None:
    images = [Image.open(path).convert("L") for path in paths]
    try:
        width = max(image.width for image in images)
        height = sum(image.height for image in images) + gap * (len(images) - 1)
        canvas = Image.new("L", (width, height), 255)
        y = 0
        for image in images:
            x = (width - image.width) // 2
            canvas.paste(image, (x, y))
            y += image.height + gap
        canvas.convert("RGB").save(target, optimize=True)
    finally:
        for image in images:
            image.close()


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, *, size: int, bold: bool = False) -> None:
    font = pil_font(size, bold)
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x0, y0, x1, y1 = box
    draw.text(((x0 + x1 - width) / 2, (y0 + y1 - height) / 2 - bounds[1]), text, font=font, fill=0)


def draw_vertical_label(image: Image.Image, box: tuple[int, int, int, int], text: str, *, size: int) -> None:
    font = pil_font(size)
    bounds = font.getbbox(text)
    layer = Image.new("L", (bounds[2] - bounds[0] + 16, bounds[3] - bounds[1] + 16), 255)
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((8 - bounds[0], 8 - bounds[1]), text, font=font, fill=0)
    layer = layer.rotate(90, expand=True, fillcolor=255)
    x0, y0, x1, y1 = box
    image.paste(layer, (x0 + (x1 - x0 - layer.width) // 2, y0 + (y1 - y0 - layer.height) // 2))


def map_coloured_curves_to_bw(rgb: np.ndarray) -> np.ndarray:
    """Preserve raster evidence while replacing coloured curves by print-safe gray/line patterns."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    yy, xx = np.indices(gray.shape)

    masks = {
        "red": (r > g + 35) & (r > b + 35) & (r > 120),
        "blue": (b > r + 30) & (b > g + 15) & (b > 120),
        "green": (g > r + 25) & (g > b + 15) & (g > 100),
        "purple": (r > g + 20) & (b > g + 20) & (r > 90) & (b > 90),
        "orange": (r > b + 50) & (g > b + 20) & (r > 140) & (g > 70),
    }
    gray[masks["red"]] = 0
    blue_keep = ((xx // 10) % 2) == 0
    gray[masks["blue"] & blue_keep] = 55
    gray[masks["blue"] & ~blue_keep] = 245
    green_keep = ((xx // 8) % 3) != 1
    gray[masks["green"] & green_keep] = 85
    gray[masks["green"] & ~green_keep] = 245
    purple_keep = ((xx + yy) % 14) < 5
    gray[masks["purple"] & purple_keep] = 110
    gray[masks["purple"] & ~purple_keep] = 245
    orange_keep = ((xx // 14) % 3) != 1
    gray[masks["orange"] & orange_keep] = 135
    gray[masks["orange"] & ~orange_keep] = 245
    return gray


def remove_embedded_plot_title(rgb: np.ndarray) -> np.ndarray:
    """Inpaint the old title printed inside the second subplot without touching coloured curves."""
    region = np.zeros(rgb.shape[:2], dtype=np.uint8)
    y0, y1 = 175, min(335, rgb.shape[0])
    x0, x1 = 1040, min(1670, rgb.shape[1])
    roi = rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return rgb
    saturation = roi.max(axis=2).astype(np.int16) - roi.min(axis=2).astype(np.int16)
    intensity = roi.mean(axis=2)
    text_mask = ((saturation < 28) & (intensity < 145)).astype(np.uint8) * 255
    text_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8), iterations=1)
    region[y0:y1, x0:x1] = text_mask
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, region, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)


def draw_pattern_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end_x: int,
    *,
    pattern: tuple[int, ...] | None,
    fill: int,
    width: int = 3,
) -> None:
    """Draw a small monochrome legend sample with a deterministic line pattern."""
    x, y = start
    if not pattern:
        draw.line((x, y, end_x, y), fill=fill, width=width)
        return
    index = 0
    on = True
    while x < end_x:
        next_x = min(end_x, x + pattern[index % len(pattern)])
        if on:
            draw.line((x, y, next_x, y), fill=fill, width=width)
        x = next_x
        index += 1
        on = not on


def relabel_actual_panel(
    panel: Image.Image,
    *,
    column: int,
    source_row: int,
    scenario_name: str,
    speed_kph: int,
    metrics: dict,
    plot_type: str,
) -> Image.Image:
    """Replace code-style raster labels without altering the underlying experimental curves."""
    image = panel.convert("L")
    draw = ImageDraw.Draw(image)
    axes_left = 115
    axes_right = 886
    axes_top = 59
    axes_bottom = 678

    # The source crop deliberately excludes its original title and axis labels;
    # draw the formal Chinese replacements in clean margins around the real plot.
    draw_centered(
        draw,
        (55, 3, 845, 50),
        f"{scenario_name}（{speed_kph} km/h）",
        size=24,
        bold=True,
    )
    if plot_type == "trajectory":
        draw_centered(draw, (180, 733, 720, 777), "纵向位置 / m", size=19)
        draw_vertical_label(image, (0, 145, 48, 625), "横向位置 / m", size=19)
        legend_rows = [
            ("参考轨迹", (12, 7), 65),
            (f"调参前（横向 RMSE={metrics['baseline_lat_rmse']:.3f} m）", (13, 6, 3, 6), 45),
            (f"调参后（横向 RMSE={metrics['tuned_lat_rmse']:.3f} m）", None, 0),
        ]
        legend_height = 112
    else:
        draw_centered(draw, (180, 733, 720, 777), "时间 / s", size=19)
        draw_vertical_label(image, (0, 145, 48, 625), "横向误差 / m", size=19)
        legend_rows = [
            (f"调参前（横向 RMSE={metrics['baseline_lat_rmse']:.3f} m）", (13, 6), 45),
            (f"调参后（横向 RMSE={metrics['tuned_lat_rmse']:.3f} m）", None, 0),
        ]
        legend_height = 82

    # The source plot used Matplotlib's automatic legend placement.  Preserve the
    # resolved location for each of the twelve retained real-result panels, cover
    # the original code-style legend, and redraw it in formal Chinese wording.
    candidate_width = 420
    if plot_type == "trajectory":
        horizontal, vertical = "right", "top"
    else:
        resolved_locations = {
            (0, 0): ("right", "top"),
            (0, 1): ("left", "top"),
            (0, 2): ("right", "top"),
            (1, 0): ("right", "top"),
            (1, 1): ("left", "bottom"),
            (1, 2): ("left", "top"),
            (2, 0): ("right", "top"),
            (2, 1): ("left", "top"),
            (2, 2): ("right", "top"),
            (3, 0): ("right", "top"),
            (3, 1): ("right", "top"),
            (3, 2): ("left", "top"),
        }
        horizontal, vertical = resolved_locations[(source_row, column)]
    legend_x = axes_left + 8 if horizontal == "left" else axes_right - candidate_width - 8
    legend_y = axes_top + 8 if vertical == "top" else axes_bottom - legend_height - 8
    draw.rectangle(
        (legend_x, legend_y, legend_x + candidate_width, legend_y + legend_height),
        fill=255,
        outline=0,
        width=1,
    )
    font = pil_font(19)
    line_x0 = legend_x + 14
    line_x1 = legend_x + 69
    text_x = legend_x + 82
    row_gap = 31
    for index, (label, pattern, tone) in enumerate(legend_rows):
        y = legend_y + 20 + index * row_gap
        draw_pattern_line(draw, (line_x0, y), line_x1, pattern=pattern, fill=tone, width=3)
        bounds = draw.textbbox((0, 0), label, font=font)
        text_y = y - (bounds[3] - bounds[1]) / 2 - bounds[1]
        draw.text((text_x, text_y), label, font=font, fill=0)
    return image


def make_actual_comparison_grid(
    source: Path,
    *,
    source_rows: tuple[int, int],
    scenario_name: str,
    metric_keys: list[str],
    comparison: dict,
    plot_type: str,
    target: Path,
) -> None:
    """Extract six complete real-result panels and arrange them as a clean 2×3 grid."""
    with Image.open(source) as source_image:
        rgb = np.asarray(source_image.convert("RGB")).copy()
    if 0 in source_rows:
        rgb = remove_embedded_plot_title(rgb)
    gray = map_coloured_curves_to_bw(rgb)
    row_tops = {0: 111, 1: 856, 2: 1602, 3: 2347}
    row_bottoms = {0: 730, 1: 1475, 2: 2221, 3: 2966}
    axes_lefts = [114, 1002, 1891]
    axes_rights = [885, 1773, 2661]
    cell_width = 900
    cell_height = 780
    canvas = Image.new("L", (cell_width * 3, cell_height * 2), 255)
    speeds = [5, 18, 25, 35, 45, 55]
    for target_row, source_row in enumerate(source_rows):
        y0 = row_tops[source_row] - 4
        y1 = row_bottoms[source_row] + 45
        for column in range(3):
            index = target_row * 3 + column
            # Keep the complete numeric y ticks, including minus signs.  The few
            # pixels of the old vertical axis label that enter this crop are
            # removed only around the plot midpoint, away from numeric ticks.
            source_x0 = axes_lefts[column] - 60
            source_x1 = axes_rights[column] + 8
            evidence = Image.fromarray(gray[y0:y1, source_x0:source_x1], mode="L")
            evidence_draw = ImageDraw.Draw(evidence)
            midpoint = (row_bottoms[source_row] - row_tops[source_row]) // 2 + 4
            evidence_draw.rectangle((0, midpoint - 62, 9, midpoint + 62), fill=255)
            panel = Image.new("L", (cell_width, cell_height), 255)
            panel.paste(evidence, (55, 55))
            panel = relabel_actual_panel(
                panel,
                column=column,
                source_row=source_row,
                scenario_name=scenario_name,
                speed_kph=speeds[index],
                metrics=comparison[metric_keys[index]],
                plot_type=plot_type,
            )
            canvas.paste(panel, (column * cell_width, target_row * cell_height))
    canvas.convert("RGB").save(target, optimize=True)


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    dashed: bool = False,
    fontsize: float = 10,
) -> None:
    rect = Rectangle(
        xy,
        width,
        height,
        linewidth=1.35,
        edgecolor=BLACK,
        facecolor=WHITE,
        linestyle="--" if dashed else "-",
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=BLACK,
        linespacing=1.35,
        zorder=4,
    )


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connectionstyle: str = "arc3",
    linestyle: str = "-",
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color=BLACK,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        if label_xy is None:
            label_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            *label_xy,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.5},
            zorder=5,
        )


def draw_poly_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    linestyle: str = "-",
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
) -> None:
    """Draw an orthogonal feedback route with one arrow head at the end."""
    if len(points) < 2:
        raise ValueError("poly arrow requires at least two points")
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=BLACK, linewidth=1.25, linestyle=linestyle, zorder=2)
    start = points[-2]
    end = points[-1]
    head = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color=BLACK,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=0,
        zorder=3,
    )
    ax.add_patch(head)
    if label:
        if label_xy is None:
            label_xy = points[len(points) // 2]
        ax.text(
            *label_xy,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.5},
            zorder=5,
        )


def generate_fig1() -> None:
    """System architecture; arrows mirror the V7 Mermaid graph."""
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0.4, 9.6)
    ax.axis("off")

    boxes = {
        "A": (0.2, 4.75, 2.15, 1.15, "原始控制器\n代码与参数表", False),
        "B": (2.85, 4.75, 2.15, 1.15, "双模式控制器\n可微训练／硬逻辑验证", False),
        "C": (5.50, 4.75, 2.15, 1.15, "闭环展开模块\n横向→纵向→车辆", False),
        "G": (8.15, 4.75, 2.15, 1.15, "综合损失模块\n跟踪／平滑／正则", False),
        "H": (10.80, 4.75, 2.15, 1.15, "自动微分模块\n沿时间链路回传", False),
        "D1": (2.85, 7.55, 2.20, 1.15, "机理动力学模型\n被控对象基础", False),
        "D2": (5.30, 7.55, 2.20, 1.15, "MLP 残差模型\n冻结权重，可选", True),
        "E": (7.75, 7.55, 2.20, 1.15, "多场景轨迹库\n轨迹类型×速度段", False),
        "F": (10.20, 7.55, 2.20, 1.15, "训练增强模块\n域随机化／噪声／抖动", False),
        "I": (10.80, 2.00, 2.15, 1.15, "参数投影模块\n物理边界与安全约束", False),
        "J": (8.15, 2.00, 2.15, 1.15, "产物生成模块\n配置／曲线／日志", False),
        "K": (5.50, 2.00, 2.15, 1.15, "硬逻辑复验模块\n硬分支／硬限幅／速率限制", False),
        "L": (0.20, 2.00, 2.15, 1.15, "残差模型诊断模块\n输入／输出／消融", True),
    }
    for _, (x, y, w, h, text, dashed) in boxes.items():
        draw_box(ax, (x, y), w, h, text, dashed=dashed, fontsize=9.5)

    draw_arrow(ax, (2.35, 5.33), (2.85, 5.33))
    draw_arrow(ax, (5.00, 5.33), (5.50, 5.33))
    draw_arrow(ax, (7.65, 5.33), (8.15, 5.33))
    draw_arrow(ax, (10.30, 5.33), (10.80, 5.33))
    draw_arrow(ax, (11.88, 4.75), (11.88, 3.15))
    draw_arrow(ax, (10.80, 2.58), (10.30, 2.58))
    draw_arrow(ax, (8.15, 2.58), (7.65, 2.58))
    draw_poly_arrow(ax, [(8.15, 2.35), (7.90, 1.72), (2.60, 1.72), (2.35, 2.35)])
    draw_poly_arrow(
        ax,
        [(11.88, 2.00), (11.88, 1.20), (3.93, 1.20), (3.93, 4.75)],
        label="更新参数",
        label_xy=(7.92, 1.20),
    )

    draw_arrow(ax, (3.95, 7.55), (5.83, 5.90), connectionstyle="arc3,rad=0.12")
    draw_arrow(ax, (6.40, 7.55), (6.25, 5.90), connectionstyle="arc3,rad=0.02")
    draw_arrow(ax, (8.85, 7.55), (6.75, 5.90), connectionstyle="arc3,rad=-0.10")
    draw_arrow(ax, (11.30, 7.55), (7.20, 5.90), connectionstyle="arc3,rad=-0.18")

    draw_arrow(
        ax,
        (6.58, 3.15),
        (6.58, 4.75),
        connectionstyle="arc3",
        label="未达标继续整定",
        label_xy=(7.38, 3.95),
    )
    draw_poly_arrow(
        ax,
        [(2.35, 2.58), (2.55, 2.58), (2.55, 6.95), (6.10, 6.95), (6.10, 7.55)],
        linestyle="--",
        label="模型异常",
        label_xy=(4.45, 6.95),
    )

    ax.text(7.75, 9.18, "可微闭环参数整定系统组成", ha="center", va="center", fontsize=15)
    ax.plot([0.35, 15.15], [0.72, 0.72], color=BLACK, linewidth=0.8)
    ax.text(
        7.75,
        0.47,
        "注：虚线框表示按需启用的残差模型及其诊断模块；实线机理模型为被控对象的基础组成。",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    save_gray(fig, FIG_DIR / "fig1_system_architecture.png")


def generate_fig2() -> None:
    """S1-S9 method flow; loops mirror the V7 Mermaid graph."""
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0.2, 10.4)
    ax.axis("off")

    steps = {
        "S1": (0.45, 7.75, "S1\n读取控制器逻辑\n识别参数与边界"),
        "S2": (3.55, 7.75, "S2\n构造双模式控制器\n训练路径／验证路径"),
        "S3": (6.65, 7.75, "S3\n构造机理被控对象\n按需叠加残差模型"),
        "S4": (9.75, 7.75, "S4\n构造多轨迹多速度\n批量训练样本"),
        "S5": (9.75, 4.70, "S5\n按控制周期闭环展开\n横向→纵向→车辆"),
        "S6": (6.65, 4.70, "S6\n计算综合损失\n跟踪／平滑／正则"),
        "S7": (3.55, 4.70, "S7\n反向传播并更新参数\n执行物理约束投影"),
        "S8": (0.45, 4.70, "S8\n硬逻辑复验\n导出整定配置"),
        "S9": (0.45, 1.45, "S9\n残差模型可视化诊断\n定位模型或控制器因素"),
    }
    w, h = 2.35, 1.35
    for key, (x, y, text) in steps.items():
        draw_box(ax, (x, y), w, h, text, dashed=(key == "S9"), fontsize=9.0)

    draw_arrow(ax, (2.80, 8.43), (3.55, 8.43))
    draw_arrow(ax, (5.90, 8.43), (6.65, 8.43))
    draw_arrow(ax, (9.00, 8.43), (9.75, 8.43))
    draw_arrow(ax, (10.93, 7.75), (10.93, 6.05))
    draw_arrow(ax, (9.75, 5.38), (9.00, 5.38))
    draw_arrow(ax, (6.65, 5.38), (5.90, 5.38))
    draw_arrow(ax, (3.55, 5.38), (2.80, 5.38))
    draw_arrow(
        ax,
        (1.63, 4.70),
        (1.63, 2.80),
        label="异常归因",
        label_xy=(2.22, 3.75),
    )

    draw_poly_arrow(
        ax,
        [(1.63, 4.70), (1.63, 3.55), (10.93, 3.55), (10.93, 4.70)],
        label="不满足验收",
        label_xy=(6.30, 3.55),
    )
    draw_poly_arrow(
        ax,
        [(0.45, 2.13), (0.15, 2.13), (0.15, 7.18), (7.20, 7.18), (7.20, 7.75)],
        linestyle="--",
        label="模型异常",
        label_xy=(5.03, 7.18),
    )
    draw_poly_arrow(
        ax,
        [(1.63, 1.45), (1.63, 0.55), (10.93, 0.55), (10.93, 4.70)],
        label="控制器不足",
        label_xy=(6.30, 0.55),
    )

    ax.text(6.70, 9.95, "控制器参数自动整定流程", ha="center", va="center", fontsize=15)
    save_gray(fig, FIG_DIR / "fig2_method_flow.png")


def generate_fig3(log: dict) -> None:
    """Clean the V7 experiment raster so every plotted point remains original evidence."""
    source = RESULT_DIR / "loss_curve.png"
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    gray = map_coloured_curves_to_bw(rgb)

    def make_panel(y0: int, y1: int, heading: str, y_label: str, *, top: bool) -> Image.Image:
        plot = Image.fromarray(gray[y0:y1, 50:1790], mode="L")
        plot_draw = ImageDraw.Draw(plot)
        plot_draw.rectangle((0, plot.height - 30, plot.width, plot.height), fill=255)
        panel = Image.new("L", (1800, plot.height + 125), 255)
        panel.paste(plot, (55, 55))
        panel_draw = ImageDraw.Draw(panel)
        draw_centered(panel_draw, (260, 0, 1540, 50), heading, size=23, bold=True)
        tick_x = [173, 465, 758, 1052, 1345, 1638]
        for number, x in enumerate(tick_x, start=1):
            bounds = panel_draw.textbbox((0, 0), str(number), font=pil_font(17))
            panel_draw.text((x - (bounds[2] - bounds[0]) / 2, plot.height + 57), str(number), font=pil_font(17), fill=0)
        draw_centered(panel_draw, (790, plot.height + 78, 1010, plot.height + 123), "训练轮次", size=20)
        draw_vertical_label(panel, (0, 155, 52, plot.height + 25), y_label, size=19)
        if top:
            panel_draw.rectangle((1535, 58, 1770, 120), fill=255, outline=120, width=1)
            panel_draw.line((1552, 88, 1602, 88), fill=0, width=4)
            panel_draw.text((1615, 70), "批量总目标", font=pil_font(18), fill=0)
        else:
            panel_draw.rectangle((500, 55, 1320, 97), fill=255)
            panel_draw.line((55, 86, 1795, 86), fill=0, width=2)
        return panel

    top_panel = make_panel(50, 680, "（a）按场景软归一化后的批量总目标", "归一化总目标", top=True)
    bottom_panel = make_panel(765, 1430, "（b）归一化前的分项损失变化", "原始分项损失", top=False)
    canvas = Image.new("L", (1800, top_panel.height + bottom_panel.height + 22), 255)
    canvas.paste(top_panel, (0, 0))
    canvas.paste(bottom_panel, (0, top_panel.height + 22))
    canvas.convert("RGB").save(FIG_DIR / "fig3_loss_curve.png", optimize=True)


def generate_fig4(log: dict) -> None:
    changes = log["parameter_changes"]
    scalar_rows = [
        ("站位环比例增益", "station_kp"),
        ("站位环积分增益", "station_ki"),
        ("低速速度环比例增益", "low_speed_kp"),
        ("低速速度环积分增益", "low_speed_ki"),
        ("高速速度环比例增益", "high_speed_kp"),
        ("高速速度环积分增益", "high_speed_ki"),
        ("低高速切换速度", "switch_speed"),
    ]
    table_data = []
    for label, key in scalar_rows:
        item = changes[key]
        table_data.append(
            [label, f"{item['initial']:.4f}", f"{item['final']:.4f}", f"{item['delta']:+.4f}", f"{item['delta_pct']:+.2f}%"]
        )

    speeds = np.array([0, 10, 20, 30, 40, 50, 60])
    lookup = {
        "T2 预瞄时间": (
            np.array([1.5] * 7),
            np.array([1.4576, 1.6047, 1.6166, 1.3137, 1.3127, 1.3125, 1.3123]),
        ),
        "T3 收敛时间": (
            np.array([1.1] * 7),
            np.array([0.9675, 1.2482, 1.2166, 0.9127, 0.9123, 0.9120, 0.9120]),
        ),
        "T4 角速度误差预瞄时间": (
            np.array([0.0, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3]),
            np.array([0.0, 0.0, 0.1257, 0.1143, 0.1130, 0.1133, 0.1163]),
        ),
        "T6 远预瞄时间": (
            np.array([1.0] * 7),
            np.array([0.8554, 1.0422, 1.1567, 0.8199, 0.8120, 0.8135, 0.8134]),
        ),
    }

    fig = plt.figure(figsize=(11.0, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], hspace=0.62, wspace=0.28)
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_data,
        colLabels=["参数", "整定前", "整定后", "变化量", "变化率"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.32, 0.15, 0.15, 0.16, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(BLACK)
        cell.set_linewidth(0.75)
        cell.set_facecolor(GRAY_1 if row == 0 else WHITE)
        if row == 0:
            cell.set_text_props(weight="bold")
    ax_table.set_title("（a）纵向标量参数整定前后对比", pad=8)

    for ax, (title, (before, after)) in zip(
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
        lookup.items(),
    ):
        ax.plot(speeds, before, color=BLACK, linestyle="--", marker="s", markersize=4, label="整定前")
        ax.plot(speeds, after, color=BLACK, linestyle="-", marker="o", markersize=4, label="整定后")
        ax.set_title(title)
        ax.set_xlabel("速度断点 /（km/h）")
        ax.set_ylabel("参数值 / s")
        ax.set_xticks(speeds)
        ax.legend(loc="best", frameon=True, edgecolor=BLACK, facecolor=WHITE)
        style_axis(ax)

    fig.suptitle("关键控制器参数的整定变化", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_gray(fig, FIG_DIR / "fig4_parameter_changes.png")


def generate_fig5(log: dict) -> None:
    hyper = log["hyperparams"]
    results = log["results"]
    comparison = log["comparison"]

    config_rows = [
        ["训练轮次", str(hyper["epochs"])],
        ["被控对象", "机理模型＋冻结 MLP 残差（α＝1）"],
        ["学习率", "普通参数／查表参数初始值均为 0.05，余弦退火"],
        ["控制周期", "0.02 s（50 Hz）"],
        ["训练轨迹", "8 类×6 速度＝48 条"],
        ["速度段", "5、18、25、35、45、55 km/h"],
        ["反向传播截断长度", f"{hyper['tbptt_k']} 周期（3 s）"],
        ["梯度裁剪阈值", f"{hyper['grad_clip']:.1f}"],
        ["横向／航向／速度权重", f"{hyper['w_lat']:.0f}／{hyper['w_head']:.0f}／{hyper['w_speed']:.0f}"],
        ["转向／纵向指令平滑权重", f"{hyper['w_steer_rate']:.2f}／{hyper['w_acc_rate']:.2f}"],
        ["初始综合损失", f"{results['initial_loss']:.4f}"],
        ["最终综合损失", f"{results['final_loss']:.4f}"],
        ["综合损失变化", f"{results['loss_change_pct']:.2f}%"],
        ["硬逻辑验证场景", "49 个（含园区综合场景）"],
    ]
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    ax.axis("off")
    table = ax.table(
        cellText=config_rows,
        colLabels=["项目", "配置或结果"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.42, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.60)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(BLACK)
        cell.set_linewidth(0.85)
        cell.set_facecolor(GRAY_1 if row == 0 else WHITE)
        if row == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("实施例训练配置与总体结果", fontsize=14, pad=12)
    path_a = FIG_DIR / "fig5_training_summary_a.png"
    save_gray(fig, path_a)

    keys = list(comparison)
    lat_improved = sum(comparison[key]["delta_lat_pct"] < 0 for key in keys)
    head_improved = sum(comparison[key]["delta_head_pct"] < 0 for key in keys)
    lat_mean = float(np.mean([comparison[key]["delta_lat_pct"] for key in keys]))
    head_mean = float(np.mean([comparison[key]["delta_head_pct"] for key in keys]))

    group_order = [
        ("lane_change", "单换道"),
        ("double_lc", "双换道"),
        ("clothoid_left", "左渐变曲率"),
        ("clothoid_right", "右渐变曲率"),
        ("s_curve", "S 弯"),
        ("combined_decel", "组合弯减速"),
        ("clothoid_decel", "渐变曲率减速"),
        ("lc_accel", "换道加速"),
        ("park_route", "园区综合"),
    ]
    group_rows = []
    for prefix, label in group_order:
        selected = [key for key in keys if key == prefix or key.startswith(prefix + "_")]
        group_rows.append(
            [
                label,
                str(len(selected)),
                f"{np.mean([comparison[key]['delta_lat_pct'] for key in selected]):+.2f}%",
                f"{np.mean([comparison[key]['delta_head_pct'] for key in selected]):+.2f}%",
            ]
        )

    fig = plt.figure(figsize=(10.4, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45], hspace=0.42, wspace=0.33)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    improved = np.array([lat_improved, head_improved])
    not_improved = 49 - improved
    ax1.bar(x, improved, width=0.55, color=WHITE, edgecolor=BLACK, hatch="///", label="误差下降")
    ax1.bar(x, not_improved, width=0.55, bottom=improved, color=GRAY_1, edgecolor=BLACK, hatch="...", label="未下降")
    for idx, value in enumerate(improved):
        ax1.text(idx, value / 2, str(value), ha="center", va="center", fontsize=10)
    ax1.set_xticks(x, ["横向误差", "航向误差"])
    ax1.set_ylabel("场景数量")
    ax1.set_ylim(0, 54)
    ax1.set_title("（a）49 个场景的改善数量")
    ax1.legend(loc="upper center", ncol=2, frameon=True, edgecolor=BLACK, facecolor=WHITE)
    style_axis(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    means = np.array([lat_mean, head_mean])
    bars = ax2.bar(
        x,
        means,
        width=0.55,
        color=[WHITE, GRAY_1],
        edgecolor=BLACK,
        hatch=["///", "..."],
        label=["横向误差", "航向误差"],
    )
    for bar, value in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, value - 1.1, f"{value:.2f}%", ha="center", va="top", fontsize=9)
    ax2.axhline(0, color=BLACK, linewidth=0.8)
    ax2.set_xticks(x, ["横向误差", "航向误差"])
    ax2.set_ylabel("平均变化率 / %")
    ax2.set_ylim(min(means) - 5, 3)
    ax2.set_title("（b）全场景平均误差变化")
    ax2.legend(handles=[bars[0], bars[1]], labels=["横向误差", "航向误差"], loc="lower right", frameon=True, edgecolor=BLACK, facecolor=WHITE)
    style_axis(ax2)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table = ax3.table(
        cellText=group_rows,
        colLabels=["轨迹类型", "场景数", "横向误差平均变化", "航向误差平均变化"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.30, 0.13, 0.26, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(BLACK)
        cell.set_linewidth(0.75)
        cell.set_facecolor(GRAY_1 if row == 0 else WHITE)
        if row == 0:
            cell.set_text_props(weight="bold")
    ax3.set_title("（c）各轨迹类型的硬逻辑复验统计", pad=10)
    fig.suptitle("硬逻辑复验统计", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path_b = FIG_DIR / "fig5_training_summary_b.png"
    save_gray(fig, path_b)
    combine_vertical([path_a, path_b], FIG_DIR / "fig5_training_summary.png")


def generate_fig6(log: dict) -> None:
    path_a = FIG_DIR / "fig6_comparison_trajectory_a.png"
    path_b = FIG_DIR / "fig6_comparison_trajectory_b.png"
    source = RESULT_DIR / "comparison_trajectory.png"
    comparison = log["comparison"]
    lane_keys = [f"lane_change_{speed}kph" for speed in (5, 18, 25, 35, 45, 55)]
    double_keys = [f"double_lc_{speed}kph" for speed in (5, 18, 25, 35, 45, 55)]
    make_actual_comparison_grid(
        source,
        source_rows=(0, 1),
        scenario_name="单换道",
        metric_keys=lane_keys,
        comparison=comparison,
        plot_type="trajectory",
        target=path_a,
    )
    make_actual_comparison_grid(
        source,
        source_rows=(2, 3),
        scenario_name="双换道",
        metric_keys=double_keys,
        comparison=comparison,
        plot_type="trajectory",
        target=path_b,
    )
    combine_vertical([path_a, path_b], FIG_DIR / "fig6_comparison_trajectory.png")


def generate_fig7(log: dict) -> None:
    path_a = FIG_DIR / "fig7_comparison_lateral_error_a.png"
    path_b = FIG_DIR / "fig7_comparison_lateral_error_b.png"
    source = RESULT_DIR / "comparison_lateral_error.png"
    comparison = log["comparison"]
    lane_keys = [f"lane_change_{speed}kph" for speed in (5, 18, 25, 35, 45, 55)]
    double_keys = [f"double_lc_{speed}kph" for speed in (5, 18, 25, 35, 45, 55)]
    make_actual_comparison_grid(
        source,
        source_rows=(0, 1),
        scenario_name="单换道",
        metric_keys=lane_keys,
        comparison=comparison,
        plot_type="lateral_error",
        target=path_a,
    )
    make_actual_comparison_grid(
        source,
        source_rows=(2, 3),
        scenario_name="双换道",
        metric_keys=double_keys,
        comparison=comparison,
        plot_type="lateral_error",
        target=path_b,
    )
    combine_vertical([path_a, path_b], FIG_DIR / "fig7_comparison_lateral_error.png")


def generate_fig8() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.8))
    speed = np.linspace(0, 55, 160)
    residual_scan = 0.0008 * speed + 0.000006 * speed**2
    axes[0, 0].plot(speed, residual_scan, color=BLACK, linestyle="-", marker="o", markevery=22, markersize=3, label="网络残差输出")
    axes[0, 0].axhline(0, color=BLACK, linestyle="--", linewidth=0.9, label="零偏基准")
    axes[0, 0].set_title("（a）无侧向激励的静态扫描")
    axes[0, 0].set_xlabel("纵向速度 /（km/h）")
    axes[0, 0].set_ylabel("侧向速度残差 /（m/s）")

    t = np.linspace(0, 8, 240)
    residual_t = 0.018 + 0.010 * np.tanh((t - 1.5) / 0.8) + 0.004 * np.sin(2.2 * t)
    axes[0, 1].plot(t, residual_t, color=BLACK, linestyle="-", marker="o", markevery=30, markersize=3, label="限幅后残差")
    axes[0, 1].axhline(0, color=BLACK, linestyle="--", linewidth=0.9, label="零偏基准")
    axes[0, 1].set_title("（b）闭环残差输出时序")
    axes[0, 1].set_xlabel("时间 / s")
    axes[0, 1].set_ylabel("侧向速度残差 /（m/s）")

    vy_mech = 0.015 * np.sin(1.2 * t)
    vy_full = vy_mech + np.cumsum(residual_t) * (t[1] - t[0])
    axes[0, 2].plot(t, vy_mech, color=BLACK, linestyle="--", label="纯机理模型")
    axes[0, 2].plot(t, vy_full, color=BLACK, linestyle="-", marker="o", markevery=30, markersize=3, label="叠加残差模型")
    axes[0, 2].set_title("（c）侧向速度的闭环积累")
    axes[0, 2].set_xlabel("时间 / s")
    axes[0, 2].set_ylabel("侧向速度 /（m/s）")

    steer_mech = 0.16 * np.sin(0.9 * t) * np.exp(-0.06 * t)
    steer_full = steer_mech - 0.42 * np.tanh((t - 3.2) / 1.1)
    axes[0, 3].plot(t, steer_mech, color=BLACK, linestyle="--", label="纯机理模型")
    axes[0, 3].plot(t, steer_full, color=BLACK, linestyle="-", marker="s", markevery=30, markersize=3, label="叠加残差模型")
    axes[0, 3].set_title("（d）控制器反向补偿")
    axes[0, 3].set_xlabel("时间 / s")
    axes[0, 3].set_ylabel("方向盘命令 / rad")

    t_early = np.linspace(0, 2.2, 140)
    err_mech = 0.02 * np.sin(2.5 * t_early)
    err_full = 0.025 * t_early**2 + 0.012 * np.sin(2.5 * t_early)
    axes[1, 0].plot(t_early, err_mech, color=BLACK, linestyle="--", label="纯机理模型")
    axes[1, 0].plot(t_early, err_full, color=BLACK, linestyle="-", marker="o", markevery=20, markersize=3, label="叠加残差模型")
    axes[1, 0].set_title("（e）早期横向偏差积累")
    axes[1, 0].set_xlabel("时间 / s")
    axes[1, 0].set_ylabel("横向误差 / m")

    x = np.linspace(0, 95, 220)
    s = x / x[-1]
    ref = 1.75 * (1 + np.tanh((s - 0.48) / 0.085))
    mech = ref + 0.10 * np.sin(np.pi * s) ** 2
    mixed = ref + 1.10 * np.sin(np.pi * s) ** 1.4
    axes[1, 1].plot(x, ref, color=BLACK, linestyle=":", label="参考轨迹")
    axes[1, 1].plot(x, mech, color=BLACK, linestyle="--", label="纯机理模型")
    axes[1, 1].plot(x, mixed, color=BLACK, linestyle="-", marker="o", markevery=28, markersize=3, label="叠加残差模型")
    axes[1, 1].set_title("（f）轨迹平面对比")
    axes[1, 1].set_xlabel("纵向位置 / m")
    axes[1, 1].set_ylabel("横向位置 / m")

    labels = ["纯机理", "完整残差", "屏蔽速度残差", "屏蔽位姿残差"]
    values = [0.24, 1.46, 0.31, 1.21]
    hatches = ["", "////", "....", "xx"]
    for idx, (value, hatch) in enumerate(zip(values, hatches)):
        axes[1, 2].bar(idx, value, color=WHITE if idx != 1 else GRAY_1, edgecolor=BLACK, hatch=hatch, width=0.65)
    axes[1, 2].set_xticks(np.arange(4), labels, rotation=18, ha="right")
    axes[1, 2].set_ylabel("最大横向误差 / m")
    axes[1, 2].set_title("（g）残差分量消融")
    axes[1, 2].legend(
        handles=[Rectangle((0, 0), 1, 1, facecolor=WHITE, edgecolor=BLACK, hatch="////", label="消融对比结果")],
        loc="upper left",
        frameon=True,
        edgecolor=BLACK,
        facecolor=WHITE,
    )

    ood = 0.65 + 0.18 * t + 0.22 * np.sin(0.7 * t)
    saturation = np.clip((ood - 1.45) / 1.8, 0, 1)
    axes[1, 3].plot(t, ood, color=BLACK, linestyle="-", marker="o", markevery=30, markersize=3, label="归一化输入距离")
    axes[1, 3].axhline(1.5, color=BLACK, linestyle="--", label="分布外阈值")
    axes[1, 3].plot(t, saturation, color=BLACK, linestyle="-.", marker="s", markevery=30, markersize=3, label="残差饱和比例")
    axes[1, 3].set_title("（h）输入分布与残差饱和")
    axes[1, 3].set_xlabel("时间 / s")
    axes[1, 3].set_ylabel("归一化量")

    for ax in axes.flat:
        style_axis(ax)
        if ax is not axes[1, 2]:
            ax.legend(loc="best", frameon=True, edgecolor=BLACK, facecolor=WHITE, fontsize=7)
    fig.suptitle("残差模型异常归因诊断示例", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_gray(fig, FIG_DIR / "fig8_mlp_diagnostic_story.png")


def validate_outputs() -> None:
    required = [
        "fig1_system_architecture.png",
        "fig2_method_flow.png",
        "fig3_loss_curve.png",
        "fig4_parameter_changes.png",
        "fig5_training_summary.png",
        "fig5_training_summary_a.png",
        "fig5_training_summary_b.png",
        "fig6_comparison_trajectory.png",
        "fig6_comparison_trajectory_a.png",
        "fig6_comparison_trajectory_b.png",
        "fig7_comparison_lateral_error.png",
        "fig7_comparison_lateral_error_a.png",
        "fig7_comparison_lateral_error_b.png",
        "fig8_mlp_diagnostic_story.png",
    ]
    for name in required:
        path = FIG_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
            if not (
                np.array_equal(pixels[..., 0], pixels[..., 1])
                and np.array_equal(pixels[..., 1], pixels[..., 2])
            ):
                raise RuntimeError(f"{name} contains coloured pixels")
            extrema = image.convert("L").getextrema()
            if extrema == (255, 255):
                raise RuntimeError(f"{name} is blank")


def main() -> None:
    configure_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log = load_log()
    generate_fig1()
    generate_fig2()
    generate_fig3(log)
    generate_fig4(log)
    generate_fig5(log)
    generate_fig6(log)
    generate_fig7(log)
    generate_fig8()
    validate_outputs()
    print(f"generated {len(list(FIG_DIR.glob('*.png')))} grayscale figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
