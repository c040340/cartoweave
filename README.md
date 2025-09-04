# CartoWeave

CartoWeave is a modern map label layout engine designed for research and rapid
experimentation. The project is split into several layers:

* a low-level **engine** containing force terms and numerical solvers,
* high-level **orchestrators** that schedule timeline-based runs, and
* lightweight **visualization** helpers for inspecting intermediate results.

Together these pieces make it easy to prototype new label placement strategies
without rebuilding common infrastructure.

## Features

* Pluggable force-based engine for point, line and area labels
* Example configurations for common layouts
* Multiple solvers: L-BFGS, semi-Newton and a hybrid strategy
* Timeline orchestrator that runs multi-stage schedules across frames
* Random scene generator for demos and tests

## Installation

```bash
pip install cartoweave          # once published
# or from source
pip install -e .
```

## Quick start

```python
from cartoweave.api import solve_frame
import numpy as np

scene = {
    "frame": 0,
    "frame_size": (1920, 1080),
    "points": np.zeros((0, 2), dtype=float),
    "lines": np.zeros((0, 4), dtype=float),
    "areas": np.zeros((0, 8), dtype=float),
    "labels_init": np.zeros((0, 2), dtype=float),
}

cfg = merge(default_cfg(), viz(show=False))
P_opt, info = solve_frame(scene, cfg)
```

Run `python examples/minimal_fit.py` for a small working demo. The example also
demonstrates the interactive viewer when ``viz.show`` is set to ``True``.

## Examples

* `examples/minimal_fit.py` – minimal example of fitting a single frame
* `examples` – more sample scripts showing how to build scenes and timelines

## Project layout

* `cartoweave/api.py` – public API for solving a single frame or a timeline
* `cartoweave/engine` – energy evaluation and numerical solvers
* `cartoweave/orchestrators` – multi-phase timeline runner
* `cartoweave/data` – random scene and timeline generators
* `cartoweave/viz` – placeholders for future visualisation tools
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

CartoWeave 是一个现代化的地图标注布局引擎，适用于科研和快速原型开发。项目由以下几部分组成：

* 底层的 **engine**，包含各种力项和数值求解器；
* 高层的 **orchestrators**，用于调度按时间线运行的多阶段流程；
* 轻量级的 **visualization** 工具，帮助检查中间结果。

这些组件组合在一起，可以让你在不重复造轮子的情况下快速尝试新的标注排版策略。

## 功能特性

* 可插拔的基于力的引擎，支持点、线、面三类标注；
* 预设的配置模板（如 `default_cfg`、`focus_only_cfg` 等），便于快速开始；
* 多种求解器：L-BFGS、半牛顿法以及混合策略；
* 按时间线运行的调度器，可执行多阶段的排版流程；
* 随机场景生成器，方便进行演示和测试。

## 安装

```bash
pip install cartoweave          # 正式发布后可直接安装
# 或者从源码安装
pip install -e .
```

## 快速上手

```python
from cartoweave.api import solve_frame
import numpy as np

scene = {
    "frame": 0,
    "frame_size": (1920, 1080),
    "points": np.zeros((0, 2), dtype=float),
    "lines": np.zeros((0, 4), dtype=float),
    "areas": np.zeros((0, 8), dtype=float),
    "labels_init": np.zeros((0, 2), dtype=float),
}

cfg = merge(default_cfg(), viz(show=False))
P_opt, info = solve_frame(scene, cfg)
```

运行 `python examples/minimal_fit.py` 可以看到一个最小示例；如果将
``viz.show`` 设为 ``True``，还能体验交互式查看器。

## 示例

* `examples/minimal_fit.py` – 拟合单帧的最小示例
* `examples` – 更多展示场景和时间线构建方式的脚本

## 项目结构

* `cartoweave/api.py` – 用于求解单帧或时间线的公共 API；
* `cartoweave/engine` – 能量评估和数值求解器；
* `cartoweave/orchestrators` – 多阶段时间线运行器；
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

