# -*- coding: utf-8 -*-
"""Minimal example using compute-only API."""
import numpy as np
from cartoweave import SolvePack, solve


def main():
    L = 1
    scene = {
        "labels_init": np.zeros((L, 2), float),
        "labels": [{"anchor_kind": "none"}]*L,
        "frame_size": (100, 100),
    }
    cfg = {"compute": {"weights": {"anchor.spring": 1.0}, "eps": {"numeric": 1e-12}}}
    sp = SolvePack(
        L=L,
        P0=scene["labels_init"],
        active_mask0=np.ones(L, dtype=bool),
        scene=scene,
        cfg=cfg,
        stages=[{"iters": 5, "solver": "lbfgs"}],
        passes=["schedule", "capture"],
    )
    view = solve(sp)
    print("[minimal_solve] final positions", view.last.P)


if __name__ == "__main__":
    main()
