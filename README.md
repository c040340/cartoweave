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

```python
import numpy as np
from cartoweave import SolvePack, solve

scene = {
    "labels_init": np.zeros((1, 2), float),
    "labels": [{"anchor_kind": "none"}],
    "frame_size": (1920, 1080),
}

cfg = {"compute": {"weights": {"anchor.spring": 1.0}, "eps": {"numeric": 1e-12}}}
sp = SolvePack(
    L=1,
    P0=scene["labels_init"],
    active_mask0=np.ones(1, dtype=bool),
    scene=scene,
    cfg=cfg,
    stages=[{"iters": 10, "solver": "lbfgs"}],
    passes=["schedule", "capture"],
)
view = solve(sp)
print(view.last.P)
```

Run `python examples/minimal_solve.py` for a small working demo.

## Command-line usage

```bash
python -m cartoweave solve --config examples/configs/compute_min.json --scene examples/scenes/scene_min.json
```


## Examples

* `examples/minimal_solve.py` – minimal example of solving a single frame

## Project layout

* `cartoweave/compute` – compute pipeline with forces, passes and solvers
* `cartoweave/data` – random scene and timeline generators
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

```python
import numpy as np
from cartoweave import SolvePack, solve

scene = {
    "labels_init": np.zeros((1, 2), float),
    "labels": [{"anchor_kind": "none"}],
    "frame_size": (1920, 1080),
}

cfg = {"compute": {"weights": {"anchor.spring": 1.0}, "eps": {"numeric": 1e-12}}}
sp = SolvePack(
    L=1,
    P0=scene["labels_init"],
    active_mask0=np.ones(1, dtype=bool),
    scene=scene,
    cfg=cfg,
    stages=[{"iters": 10, "solver": "lbfgs"}],
    passes=["schedule", "capture"],
)
view = solve(sp)
print(view.last.P)
```

运行 `python examples/minimal_fit.py` 可以看到一个最小示例；如果将
``viz.show`` 设为 ``True``，还能体验交互式查看器。

## 示例

* `examples/minimal_fit.py` – 拟合单帧的最小示例
* `examples` – 更多展示场景和时间线构建方式的脚本

## 项目结构

* `cartoweave/compute` – 核心计算管线，包含力项、passes 和求解器；
* `cartoweave/data` – 随机场景和时间线生成器；
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

