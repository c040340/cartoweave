"""Run a minimal solve using the config-driven API."""

from cartoweave.compute import solve
from cartoweave.data.api import build_solvepack_from_config

cfg = {
    "data": {
        "source": "generate",
        "generate": {
            "frame_size": [200, 100],
            "num_labels": 3,
            "num_points": 1,
            "num_lines": 1,
            "num_areas": 1,
            "num_steps": 3,
            "seed": 0,
        },
    }
}

sp = build_solvepack_from_config(cfg, seed=0)
sp.validate()
view = solve(sp)
print("frames:", view.summary.get("frames_captured"), "E_last:", view.summary.get("E_last"))
