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

The result contains coordinates for the last frame of each action only.

```python
from cartoweave import solve_layout

labels = [
    {"id": 0, "WH": [80, 24], "anchor": {"kind": "point", "id": 0}},
]
elements = {"points": [{"id": 0, "xy": [100.0, 200.0]}]}
actions = [{"t": 0, "op": "activate", "target": "label", "ids": "all"}]

cfg = {
    "compute": {"public": {"forces": {"anchor.spring": {"enable": True, "k_local": 1.0}}}},
}
res = solve_layout(labels, elements, actions, config_profile=cfg)
print(res.coords[-1])  # final coordinates after the action
```

Run `python examples/minimal_solve.py` for a small working demo.

Each element (point, polyline or polygon) should provide an ``id`` so that
labels can reference it via their ``anchor`` field.

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

### Solver stability defaults
- We keep L-BFGS strictly smooth by default:
  - `step_limit` **enabled** (default `max_step_norm: 1.5`)
  - `grad_clip` **disabled** (turn on only for debugging/stability)
- `area.cross` now uses a **continuous gate** with AABB pre‑clipping and
  short‑edge skipping for robust collision handling.

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
    {"id": 0, "WH": [80, 24], "anchor": {"kind": "point", "id": 0}},
]
elements = {"points": [{"id": 0, "xy": [100.0, 200.0]}]}
actions = [{"t": 0, "op": "activate", "target": "label", "ids": "all"}]

cfg = {
    "compute": {"public": {"forces": {"anchor.spring": {"enable": True, "k_local": 1.0}}}},
}
res = solve_layout(labels, elements, actions, config_profile=cfg)
print(res.coords[-1])
```

运行 `python examples/minimal_solve.py` 可以看到一个最小示例；如果将
``viz.show`` 设为 ``True``，还能体验交互式查看器。

每个场景元素（点、线、面）需提供 ``id``，并由标注通过 ``anchor`` 字段关联。

## 示例

* `examples/minimal_solve.py` – 拟合单帧的最小示例
* `examples` – 更多展示场景和步骤配置方式的脚本

### 求解器稳定性默认值
- 默认情况下保持 L-BFGS 完全平滑：
  - `step_limit` **启用**（默认 `max_step_norm: 1.5`）
  - `grad_clip` **禁用**（仅在调试/稳定性需要时开启）
- `area.cross` 现在使用 **连续门函数**，并加入 AABB 预裁剪与短边跳过，
  提供更稳健的相交惩罚。

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

