# CartoWeave

CartoWeave is a modern map label layout engine. It separates the project into a low-level **engine** of force terms and solvers, high-level **orchestrators** for timeline-based runs, and placeholder **visualization** helpers.

## Features

- Pluggable force-based engine for point, line, and area labels
- Config presets for common layouts (`default_cfg`, `focus_only_cfg`, etc.)
- Multiple solvers: L-BFGS, semi-Newton, and a hybrid strategy
- Timeline orchestrator that runs multi-stage schedules
- Random scene generator for demos and tests

## Installation

```bash
pip install cartoweave          # once published
# or from source
pip install -e .
```

## Quick start

```python
from cartoweave.api import solve_frame
from cartoweave.config.presets import default_cfg
from cartoweave.config.utils import merge, viz
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

Run `python examples/minimal_fit.py` for a tiny working demo that also
demonstrates the interactive viewer when ``viz.show`` is set to ``True``.

## Project layout

- `cartoweave/api.py` – public API for solving a single frame or a timeline
- `cartoweave/config` – utilities and presets for force constants
- `cartoweave/engine` – energy evaluation and numerical solvers
- `cartoweave/orchestrators` – multi-phase timeline runner
- `cartoweave/data` – random scene and timeline generators
- `cartoweave/viz` – placeholders for future visualisation tools

## Testing

```bash
pytest
```

## License

MIT
