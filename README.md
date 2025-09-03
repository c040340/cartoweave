# CartoWeave

A modern map label layout engine for points, lines and areas — with a clean Engine / Orchestrator / Viz split.

## Quick start
```python
from cartoweave.api import solve_frame
import numpy as np

scene = {
    "frame": 0,
    "frame_size": (1920, 1080),
    "points": np.zeros((0, 2), dtype=float),
    "lines": np.zeros((0, 4), dtype=float),  # x1,y1,x2,y2 per row
    "areas": np.zeros((0, 8), dtype=float),  # polygon packed (placeholder)
    "labels_init": np.zeros((0, 2), dtype=float),
}
P_opt, info = solve_frame(scene, cfg={})
