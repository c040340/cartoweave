# CartoWeave

CartoWeave is a modern map label layout engine designed for research and rapid
experimentation. The legacy "engine" and "orchestrators" modules have been
retired in favour of a compute-only pipeline with pluggable passes and solvers.
Lightweight visualization helpers remain for inspecting intermediate results.

This new structure makes it easy to prototype new label placement strategies
without rebuilding common infrastructure.

## Features

* Pluggable force terms for point, line and area labels
* Example configurations for common layouts
* Multiple solvers: L-BFGS and semi-Newton
* Pass manager with scheduling, clipping and capture hooks
* Random scene generator for demos and tests

## Installation

```bash
pip install cartoweave          # once published
# or from source
pip install -e .
```

## Quick start

The result contains coordinates for the last frame of each action only. The
`frame_size=(W, H)` argument is required and must match your scene coordinates.

```python
from cartoweave import solve_layout

labels = [
    {"label_id": 0, "WH": [80, 24], "anchor": {"kind": "point", "element_id": 0}},
]
elements = {"points": [{"element_id": 0, "xy": [100.0, 200.0]}]}
actions = [{"t": 0, "op": "activate", "label_ids": "all"}]

cfg = {
    "compute": {"public": {"forces": {"anchor.spring": {"enable": True, "k_local": 1.0}}}},
}
res = solve_layout(labels, elements, actions, frame_size=(1280, 800), config_profile=cfg)
print(res.coords[-1])  # final coordinates after the action
```

Run `python examples/minimal_solve.py` for a small working demo. For running a
specific call from a payload JSON and opening the interactive viz, try:

```bash
python examples/run_payload_shot_viz.py --file cartoweave_payload_latest.json --shot 2 --frame 1280 800 --viz
```

Each element (point, polyline or polygon) should provide an ``element_id`` so
that labels can reference it via their ``anchor`` field. Labels themselves use
``label_id`` and actions reference labels via ``label_ids``.

## Data API

```python
from cartoweave.data.api import (
    build_solvepack_from_config,
    load_solvepack_from_file,
)
```

Canonical YAML snippet:

```yaml
data:
  source: random
  path: ""
  regen: false

  frame: { width: 1280, height: 800 }

  counts:
    labels: 16
    points: 6
    lines: 4
    areas: 2

  random:
    route_gen:
      segment_len_scale: 0.06
      inset_margin_scale: 0.05
      min_vertex_spacing_scale: 0.03
    area_gen:
      inset_margin_scale: 0.06
      min_edge_spacing_scale: 0.03

  anchors:
    policy: auto
    modes:
      line: projected
      area: projected_edge

  steps:
    kind: sequential
    steps: 16
```

### Steps

- `steps.kind="none"` ⇒ single stage (no actions).
- `steps.kind="sequential"`:
  - if `steps == labels` ⇒ one label per step;
  - if `steps < labels` ⇒ first `steps-1` add one label each; last stage activates all remaining.
- `steps.kind="grouped"`: via `group_sizes` or explicit `groups`.

## Command-line usage

```bash
python -m cartoweave solve --config examples/configs/compute_min.json --scene examples/scenes/scene_min.json
```


## Examples

* `examples/minimal_solve.py` – minimal example of solving a single frame

### Solver stability & tuning
- We ship a conservative pass setup by default:
  - `step_limit` enabled (default `max_step_norm: 60`) to cap per‑step motion.
  - `grad_clip` enabled (default `max_norm: 1000.0`) to tame extreme gradients.
  - You can raise `grad_clip.max_norm` or disable it if your first pass needs a
    stronger initial push; keep `step_limit` to prevent overshoot.
- `appear_nudge` gives newly activated labels a gentle initial displacement; you
  can raise `step_px` (e.g. 4–8) for clearer “push‑off” in the first pass.
- `area.cross` uses a continuous gate with AABB pre‑clipping and short‑edge
  skipping for robust collision handling.

### Label–Label kernels
- Rectangle kernel: `ll.rect` (anisotropic, respects W×H)
- Disk/Ellipse kernel: `ll.disk`
  - `ll_disk_mode: max|min|avg|diag|ellipse`
  - `ellipse` uses directional support radii of axis‑aligned ellipses for a
    more faithful interaction with wide/flat labels.

### Anchors
- `anchor.spring` acts on point/line anchors; area‑anchored labels are
  constrained by `area.*` terms (`area.embed`, `area.cross`, `area.softout`).

## Project layout

* `cartoweave/compute` – compute pipeline with forces, passes and solvers
* `cartoweave/data` – random scene utilities and data API
* `cartoweave/viz` – placeholders for visualisation tools
* `tests` – unit tests and integration tests

## Testing

```bash
pytest
```

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0). You may copy and modify the code for non-commercial
purposes, provided you give appropriate credit and distribute your
contributions under the same license. See the LICENSE file for details.

---

# CartoWeave（中文）

CartoWeave 是一个现代化的地图标注布局引擎，适用于科研和快速原型开发。旧版的
engine/orchestrators 模块已被移除，现在的框架专注于 compute 管线和可插拔的
求解 passes，仍保留轻量级的可视化工具。

## 功能特性

* 可插拔的力项，支持点、线、面三类标注；
* 预设的配置模板，便于快速开始；
* 多种求解器：L-BFGS、半牛顿法；
* PassManager 提供调度、裁剪和捕获等扩展点；
* 随机场景生成器，方便进行演示和测试。

## 安装

```bash
pip install cartoweave          # 正式发布后可直接安装
# 或者从源码安装
pip install -e .
```

## 快速上手

返回结果仅包含每个动作最终一步的坐标。

```python
from cartoweave import solve_layout

labels = [
    {"label_id": 0, "WH": [80, 24], "anchor": {"kind": "point", "element_id": 0}},
]
elements = {"points": [{"element_id": 0, "xy": [100.0, 200.0]}]}
actions = [{"t": 0, "op": "activate", "label_ids": "all"}]

cfg = {
    "compute": {"public": {"forces": {"anchor.spring": {"enable": True, "k_local": 1.0}}}},
}
res = solve_layout(labels, elements, actions, frame_size=(1280, 800), config_profile=cfg)
print(res.coords[-1])
```

运行 `python examples/minimal_solve.py` 可以看到一个最小示例；如果将
``viz.show`` 设为 ``True``，还能体验交互式查看器。

每个场景元素（点、线、面）需提供 ``element_id``，标注使用 ``label_id``，
动作通过 ``label_ids`` 指定作用的标注。

## 示例

* `examples/minimal_solve.py` – 拟合单帧的最小示例
* `examples/solve_layout_api.py` – 展示外部 `solve_layout` 接口处理点、线、面场景并包含多动作
* `examples` – 更多展示场景和步骤配置方式的脚本

### 求解器稳定性与调参
- 默认包含以下保护：
  - `step_limit` 启用（默认 `max_step_norm: 60`）限制单步位移；
  - `grad_clip` 启用（默认 `max_norm: 1000.0`）裁剪大梯度；
  - 若首帧需要更强的“推开”，可调大 `grad_clip.max_norm` 或临时关闭，同时保留
    `step_limit`；也可调大 `appear_nudge.step_px`（如 4–8）。
- `area.cross` 使用连续门函数 + AABB 预裁剪 + 短边跳过，更稳健。

### 标签-标签核
- 矩形核：`ll.rect`（各向异性，遵循宽高）
- 圆/椭圆核：`ll.disk`
  - `ll_disk_mode: max|min|avg|diag|ellipse`
  - `ellipse` 使用轴对齐椭圆在方向上的支持函数，更适合“又宽又扁”的标签。

### 锚点
- `anchor.spring` 作用于点/线锚；面锚由 `area.*`（`area.embed`、`area.cross`、
  `area.softout`）约束。

## 项目结构

* `cartoweave/compute` – 核心计算管线，包含力项、passes 和求解器；
* `cartoweave/data` – 随机场景与数据 API；
* `cartoweave/viz` – 计划中的可视化工具；
* `tests` – 单元测试与集成测试。

## 测试

```bash
pytest
```

## 许可证

本项目采用 Creative Commons 署名-非商业性使用-相同方式共享 4.0 国际
（CC BY-NC-SA 4.0）许可证。您可以在非商业目的下复制和修改代码，但必须标注
作者并以相同的协议共享您的贡献。详见 LICENSE 文件。
