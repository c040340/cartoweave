"""Minimal example that generates a scene using the new data API."""
from __future__ import annotations

import numpy as np

from cartoweave.data.api import build_solvepack_from_config


def main() -> None:
    cfg = {
        "data": {
            "source": "generate",
            "generate": {
                "frame_size": [200, 100],
                "num_labels": 5,
                "num_points": 2,
                "num_lines": 2,
                "num_areas": 1,
                "num_steps": 5,
                "seed": 0,
            },
        }
    }
    sp = build_solvepack_from_config(cfg, seed=0)
    sp.validate()
    scene = sp.scene0
    print("frame", scene.frame_size)
    print("labels", sp.L)


if __name__ == "__main__":
    main()
