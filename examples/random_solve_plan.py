"""Minimal example that generates a scene using the new data API."""
from __future__ import annotations

import numpy as np
from cartoweave.data.api import build_solvepack_direct


def main() -> None:
    sp = build_solvepack_direct(
        frame_size=(200, 100),
        n_labels=5,
        n_points=2,
        n_lines=2,
        n_areas=1,
        steps={"kind": "sequential", "steps": 5},
        seed=0,
    )
    scene = sp.scene
    print("points", np.asarray(scene["points"]).shape)
    print("lines", len(scene["lines"]))
    print("areas", len(scene["areas"]))
    print("labels", sp.L)


if __name__ == "__main__":
    main()
