import numpy as np
import pytest
from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def test_steps_group_sizes_capture(as_legacy_pack):
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 5, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }
    sp = as_legacy_pack(build_solvepack_from_config(cfg, seed=0))
    vp = solve(sp)
    assert isinstance(vp.frames, list) and len(vp.frames) >= 1
    scene = getattr(sp, "scene0", None)
    if scene is None:
        scene = sp.scene
    labels = scene.labels if hasattr(scene, "labels") else scene["labels"]
    assert len(labels) == sp.N


def test_anchors_modes_passthrough():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 0, "num_lines": 1, "num_areas": 1, "num_steps": 1},
        },
        "behavior": {
            "anchor_policy": "round_robin",
            "default_WH": {"point": [8, 8], "line": [12, 6], "area": [40, 30]},
            "requested_modes": {"line": "midpoint", "area": "centroid"},
        },
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }
    sp = build_solvepack_from_config(cfg, seed=1)
    labels = sp.scene0.labels
    allowed = {"none", "centroid", "midpoint"}
    assert all(l.get("anchor_kind", "none") in {"none", "line", "area"} for l in labels)
    assert all(("mode" not in l) or (l["mode"] in allowed) for l in labels)
