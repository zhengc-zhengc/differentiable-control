"""Build V5 patent disclosure with formula numbers and clean subheadings."""

from __future__ import annotations

import importlib.util
import re
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
TMP_DIR = OUT_DIR / "_tmp"
V3_DIR = ROOT / "专利撰写" / "可微控制参数整定交底书_V3"
V3_FIG_DIR = V3_DIR / "figures"
CASE_NAME = "一种基于车辆动力学可微闭环仿真的车辆横纵向控制器参数自动整定方法"


def load_v3_module():
    spec = importlib.util.spec_from_file_location("build_disclosure_v3", V3_DIR / "build_disclosure_v3.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load V3 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V3 = load_v3_module()


def arrowhead(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex >= sx else -1
        pts = [(ex, ey), (ex - sign * 14, ey - 8), (ex - sign * 14, ey + 8)]
    else:
        sign = 1 if ey >= sy else -1
        pts = [(ex, ey), (ex - 8, ey - sign * 14), (ex + 8, ey - sign * 14)]
    draw.polygon(pts, fill="black")


def poly_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]]) -> None:
    draw.line(points, fill="black", width=2)
    arrowhead(draw, points[-2], points[-1])


def straight_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    poly_arrow(draw, [start, end])


def center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def generate_system_diagram(path: Path) -> None:
    img = Image.new("RGB", (1800, 1000), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 35), "图1  可微闭环整定系统框图", fill="black", font=V3.font(34, bold=True))

    boxes = {
        "A": (70, 330, 350, 450, "原始控制器\n代码/参数表"),
        "B": (430, 330, 710, 450, "双模式控制器\n训练路径/验证路径"),
        "C": (790, 330, 1080, 450, "闭环展开模块\n横向-纵向-动力学"),
        "D": (1160, 330, 1450, 450, "综合损失模块\n误差/平滑/正则"),
        "E": (1510, 330, 1740, 450, "自动微分模块\n梯度回传"),
        "F": (690, 110, 940, 230, "车辆动力学模型\n名义/参数域"),
        "G": (970, 110, 1220, 230, "多场景轨迹库\n轨迹/速度段"),
        "H": (1240, 110, 1490, 230, "训练增强模块\n域随机化/噪声/抖动"),
        "I": (1160, 620, 1450, 750, "参数投影模块\n物理边界约束"),
        "J": (790, 620, 1080, 750, "产物生成模块\n整定配置/日志"),
        "K": (430, 620, 710, 750, "硬逻辑复验模块\n原始分支/限幅/速率"),
    }
    highlight = {"B", "C", "D", "E", "I"}
    for key, (x1, y1, x2, y2, text) in boxes.items():
        V3.draw_box(draw, (x1, y1, x2, y2), text, fill="#F7F7F7" if key in highlight else "white")

    # Main forward chain.
    straight_arrow(draw, (350, 390), (430, 390))
    straight_arrow(draw, (710, 390), (790, 390))
    straight_arrow(draw, (1080, 390), (1160, 390))
    straight_arrow(draw, (1450, 390), (1510, 390))

    # Orthogonal input connections into the closed-loop and loss path.
    straight_arrow(draw, (815, 230), (815, 330))
    poly_arrow(draw, [(1095, 230), (1095, 280), (1015, 280), (1015, 330)])
    straight_arrow(draw, (1365, 230), (1365, 330))

    # Parameter update, output, validation, and outer feedback loop.
    poly_arrow(draw, [(1625, 450), (1625, 535), (1305, 535), (1305, 620)])
    straight_arrow(draw, (1160, 685), (1080, 685))
    straight_arrow(draw, (790, 685), (710, 685))
    poly_arrow(draw, [(570, 750), (570, 870), (745, 870), (745, 535), (935, 535), (935, 450)])
    draw.text((575, 835), "复验未达标则继续整定", fill="black", font=V3.font(20))

    draw.text(
        (70, 930),
        "说明：训练路径用于获得参数更新方向，验证路径保留工程硬逻辑用于部署前复验。",
        fill="black",
        font=V3.font(18),
    )
    img.save(path)


def generate_flow_diagram(path: Path) -> None:
    img = Image.new("RGB", (1800, 1050), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 35), "图2  控制器参数自动整定流程图", fill="black", font=V3.font(34, bold=True))

    w, h = 310, 125
    y_top, y_bottom = 155, 450
    xs = [100, 525, 950, 1375]
    steps = {
        "S1": (xs[0], y_top, xs[0] + w, y_top + h, "S1\n读取控制器逻辑\n识别参数与边界"),
        "S2": (xs[1], y_top, xs[1] + w, y_top + h, "S2\n构造双模式控制器\n可微训练/硬逻辑验证"),
        "S3": (xs[2], y_top, xs[2] + w, y_top + h, "S3\n构造车辆动力学\n建立物理参数域"),
        "S4": (xs[3], y_top, xs[3] + w, y_top + h, "S4\n构造多轨迹多速度\n批量闭环样本"),
        "S5": (xs[3], y_bottom, xs[3] + w, y_bottom + h, "S5\n按控制周期展开\n横向-纵向-动力学"),
        "S6": (xs[2], y_bottom, xs[2] + w, y_bottom + h, "S6\n计算综合损失\n误差/平滑/正则"),
        "S7": (xs[1], y_bottom, xs[1] + w, y_bottom + h, "S7\n反向传播更新\n执行物理投影"),
        "S8": (xs[0], y_bottom, xs[0] + w, y_bottom + h, "S8\n硬逻辑复验\n导出整定配置"),
    }
    for key, (x1, y1, x2, y2, text) in steps.items():
        V3.draw_box(draw, (x1, y1, x2, y2), text, title=key in {"S1", "S8"}, fill="#F7F7F7")

    straight_arrow(draw, (steps["S1"][2], center(steps["S1"][:4])[1]), (steps["S2"][0], center(steps["S2"][:4])[1]))
    straight_arrow(draw, (steps["S2"][2], center(steps["S2"][:4])[1]), (steps["S3"][0], center(steps["S3"][:4])[1]))
    straight_arrow(draw, (steps["S3"][2], center(steps["S3"][:4])[1]), (steps["S4"][0], center(steps["S4"][:4])[1]))
    straight_arrow(draw, (center(steps["S4"][:4])[0], steps["S4"][3]), (center(steps["S5"][:4])[0], steps["S5"][1]))
    straight_arrow(draw, (steps["S5"][0], center(steps["S5"][:4])[1]), (steps["S6"][2], center(steps["S6"][:4])[1]))
    straight_arrow(draw, (steps["S6"][0], center(steps["S6"][:4])[1]), (steps["S7"][2], center(steps["S7"][:4])[1]))
    straight_arrow(draw, (steps["S7"][0], center(steps["S7"][:4])[1]), (steps["S8"][2], center(steps["S8"][:4])[1]))

    poly_arrow(
        draw,
        [
            (center(steps["S8"][:4])[0], steps["S8"][3]),
            (center(steps["S8"][:4])[0], 850),
            (center(steps["S5"][:4])[0], 850),
            (center(steps["S5"][:4])[0], steps["S5"][3]),
        ],
    )
    draw.text((760, 815), "复验未达标则继续整定", fill="black", font=V3.font(20))
    img.save(path)


def generate_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    generate_system_diagram(FIG_DIR / "fig1_system_architecture.png")
    generate_flow_diagram(FIG_DIR / "fig2_method_flow.png")
    for name in [
        "fig3_loss_curve.png",
        "fig4_parameter_changes.png",
        "fig5_training_summary.png",
        "fig6_comparison_trajectory.png",
        "fig7_comparison_lateral_error.png",
    ]:
        V3.grayscale_copy(V3_FIG_DIR / name, FIG_DIR / name)


SYSTEM_MERMAID = r"""
```mermaid
flowchart LR
  A["原始控制器代码/参数表"] --> B["双模式控制器"]
  B --> C["闭环展开模块"]
  D["车辆动力学模型"] --> C
  E["多场景轨迹库"] --> C
  F["训练增强模块：域随机化/噪声/抖动"] --> C
  C --> G["综合损失模块"]
  G --> H["自动微分模块"]
  H --> I["参数投影模块"]
  I --> J["产物生成模块"]
  J --> K["硬逻辑复验模块"]
  K --"未达标继续整定"--> C
```
<!-- ![图1 可微闭环整定系统框图](figures/fig1_system_architecture.png) -->
"""


FLOW_MERMAID = r"""
```mermaid
flowchart TB
  S1["S1 读取控制器逻辑并识别参数"] --> S2["S2 构造双模式控制器"]
  S2 --> S3["S3 构造车辆动力学与参数域"]
  S3 --> S4["S4 构造多轨迹多速度训练集"]
  S4 --> S5["S5 按控制周期闭环展开"]
  S5 --> S6["S6 计算综合损失"]
  S6 --> S7["S7 反向传播并投影更新参数"]
  S7 --> S8["S8 硬逻辑复验并导出配置"]
  S8 --"不满足验收"--> S5
```
<!-- ![图2 控制器参数自动整定流程](figures/fig2_method_flow.png) -->
"""


def replace_section(md: str, start: str, end: str, replacement: str) -> str:
    pattern = re.escape(start) + r".*?(?=" + re.escape(end) + r")"
    updated, count = re.subn(pattern, replacement, md, flags=re.S)
    if count != 1:
        raise RuntimeError(f"expected to replace one section starting with {start!r}, got {count}")
    return updated


def formalize_markdown(md: str) -> str:
    md = re.sub(r"\n\*\*版本说明\*\*：.*?\n", "\n", md)
    md = re.sub(r"\n---\n\n## 注意事项\n.*?\n## 一、", "\n---\n\n## 一、", md, flags=re.S)

    replacements = {
        "## 一、介绍相关技术背景，描述与本发明技术最相近的现有技术，并说明该现有技术存在的缺点": "## 一、技术背景、最接近现有技术及现有技术缺点",
        "## 二、针对上述缺点，说明本发明所要解决的技术问题": "## 二、本发明所要解决的技术问题",
        "## 四、与现有技术相比，本发明具有哪些优点？": "## 四、与现有技术相比的有益效果",
        "## 五、本发明的技术关键点和欲保护点是什么？": "## 五、本发明的技术关键点和保护点",
        "## 六、其它（实施例、技术效果、参数示例）": "## 六、实施例、技术效果和参数示例",
    }
    for old, new in replacements.items():
        md = md.replace(old, new)

    md = md.replace(V3.SYSTEM_MERMAID.strip(), SYSTEM_MERMAID.strip())
    md = md.replace(V3.FLOW_MERMAID.strip(), FLOW_MERMAID.strip())
    md = md.replace("可部署 YAML 参数配置", "可部署参数配置")
    md = md.replace(
        "图6和图7示出了代表场景的轨迹跟踪和横向误差对比；完整结果保留在项目结果目录中。",
        "图6和图7示出了代表场景的轨迹跟踪和横向误差对比；完整验证材料可随整定日志一并保存。",
    )
    md = md.replace(" kph", " km/h")

    section2 = """## 二、本发明所要解决的技术问题

本发明所要解决的技术问题在于，在不改变既有车辆横向、纵向工程控制器主体结构的前提下，构建能够用于参数自动整定的车辆动力学可微闭环仿真环境，使控制器参数能够根据轨迹跟踪误差、速度误差和控制平滑性指标自动更新。

进一步地，本发明还解决训练可微性与工程可部署性之间的冲突：训练阶段允许对限幅、分段切换、速率限制等非光滑环节采用可微近似，以便获得参数更新方向；验证和部署前仍使用原始硬限幅、硬分支和硬速率限制进行闭环复验，以保证整定参数能够回到原工程控制模块中使用。

此外，本发明还解决多轨迹、多速度和车辆物理参数不确定性下的统一整定问题，使横向预瞄、收敛、角速度误差预瞄、纵向位置环和速度环等多类参数能够在同一批量闭环训练过程中协同更新，并通过物理边界约束和日志产物保证整定过程可追溯。

"""
    md = replace_section(md, "## 二、本发明所要解决的技术问题", "## 三、本发明技术方案的详细阐述", section2)

    md = md.replace(
        "## 四、与现有技术相比的有益效果\n\n（1）",
        "## 四、与现有技术相比的有益效果\n\n本发明相较于现有技术至少具有以下有益效果。\n\n（1）",
    )
    md = md.replace(
        "## 五、本发明的技术关键点和保护点\n\n（1）",
        "## 五、本发明的技术关键点和保护点\n\n本发明建议重点保护以下技术方案和技术特征。\n\n（1）",
    )

    section64 = """### 6.4 可实施性说明

上述实施例可以采用通用计算设备、仿真服务器或车端开发环境实施。实施时，将车辆横向控制逻辑、纵向控制逻辑、车辆动力学模型、训练场景库、参数边界表和验证场景库配置在同一整定系统中；训练完成后，系统输出与原控制器参数格式一致或可转换为原参数格式的整定配置。

在一个可选实施方式中，系统保存每轮训练的参数变化、损失变化、代表场景跟踪曲线、硬逻辑复验统计和配置导出日志。上述材料用于说明参数整定过程、复验结果和部署前差异，不限定本发明的具体软件目录、编程语言、配置文件名称或图表样式。
"""
    if "### 6.4 项目实现依据" in md:
        md = md[: md.index("### 6.4 项目实现依据")] + section64
    return add_equation_numbers(md).strip() + "\n"


def add_equation_numbers(md: str) -> str:
    """Add visible equation numbers to the displayed formulas cited in text."""
    replacements = {
        r"""
\[
u_t = g(x_t,r_t,\theta)
\]
""".strip(): r"""
\[
u_t = g(x_t,r_t,\theta) \qquad \mathrm{(1)}
\]
""".strip(),
        r"""
\[
x_{t+1} = f(x_t,u_t,\phi)
\]
""".strip(): r"""
\[
x_{t+1} = f(x_t,u_t,\phi) \qquad \mathrm{(2)}
\]
""".strip(),
        r"""
\[
J(\theta)=\sum_{t=0}^T\left(w_{\mathrm{lat}}e_{t,\mathrm{lat}}^2+w_{\mathrm{head}}e_{t,\mathrm{head}}^2+w_{\mathrm{spd}}e_{t,\mathrm{spd}}^2+w_{\mathrm{smooth}}\lVert\Delta u_t\rVert^2\right)+w_{\mathrm{reg}}\lVert\theta-\theta_0\rVert^2
\]
""".strip(): r"""
\[
\begin{aligned}
J(\theta)=&\sum_{t=0}^T\left(w_{\mathrm{lat}}e_{t,\mathrm{lat}}^2+w_{\mathrm{head}}e_{t,\mathrm{head}}^2+w_{\mathrm{spd}}e_{t,\mathrm{spd}}^2+w_{\mathrm{smooth}}\lVert\Delta u_t\rVert^2\right)\\
&+w_{\mathrm{reg}}\lVert\theta-\theta_0\rVert^2 \qquad \mathrm{(3)}
\end{aligned}
\]
""".strip(),
        r"""
\[
\theta^{k+1}=\Pi_{\Theta}\left(\theta^k-\eta\nabla_{\theta}J(\theta^k)\right)
\]
""".strip(): r"""
\[
\theta^{k+1}=\Pi_{\Theta}\left(\theta^k-\eta\nabla_{\theta}J(\theta^k)\right) \qquad \mathrm{(4)}
\]
""".strip(),
        r"""
\[
M_{\mathrm{hard}}(\theta^*) \rightarrow \{\mathrm{trajectory},\mathrm{error},\mathrm{command},\mathrm{log}\}
\]
""".strip(): r"""
\[
M_{\mathrm{hard}}(\theta^*) \rightarrow \{\mathrm{trajectory},\mathrm{error},\mathrm{command},\mathrm{log}\} \qquad \mathrm{(5)}
\]
""".strip(),
    }
    for old, new in replacements.items():
        md = md.replace(old, new)
    return md


def disclosure_md(timestamp: str) -> str:
    return formalize_markdown(V3.disclosure_md(timestamp))


def docx_markdown(md: str) -> str:
    md = md.replace(SYSTEM_MERMAID.strip(), "![图1 可微闭环整定系统框图](figures/fig1_system_architecture.png)")
    md = md.replace(FLOW_MERMAID.strip(), "![图2 控制器参数自动整定流程](figures/fig2_method_flow.png)")
    return md


def run_pandoc(src_md: Path, out_docx: Path) -> None:
    cmd = [
        "pandoc",
        str(src_md),
        "-o",
        str(out_docx),
        "--from",
        "markdown+tex_math_dollars+tex_math_single_backslash",
        "--resource-path",
        str(OUT_DIR),
        "--metadata",
        f"title={CASE_NAME}",
    ]
    subprocess.run(cmd, check=True, cwd=OUT_DIR)


def clear_heading_italics(path: Path) -> None:
    from docx import Document

    doc = Document(path)
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            for run in para.runs:
                run.italic = False
    doc.save(path)


def build(timestamp: str | None = None) -> tuple[Path, Path, dict[str, int]]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    generate_figures()
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    base = f"{V3.safe_name(CASE_NAME)}_{timestamp}"
    md_path = OUT_DIR / f"{base}.md"
    docx_path = OUT_DIR / f"{base}.docx"
    tmp_md = TMP_DIR / f"{base}.docx.md"

    md = disclosure_md(timestamp)
    md_path.write_text(md, encoding="utf-8")
    tmp_md.write_text(docx_markdown(md), encoding="utf-8")

    run_pandoc(tmp_md, docx_path)
    V3.style_docx(docx_path)
    clear_heading_italics(docx_path)
    stats = V3.patch_docx_ooxml(docx_path)
    return md_path, docx_path, stats


if __name__ == "__main__":
    md, docx, stats = build()
    print(f"wrote {md}")
    print(f"wrote {docx}")
    print(f"stats {stats}")
