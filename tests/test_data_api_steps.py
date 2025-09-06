import numpy as np
from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def _basic_cfg():
    return {
        "data": {
            "source": "generate",
            "generate": {"num_points": 3, "num_lines": 1, "num_areas": 0, "num_steps": 1},
        },
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }


def test_sequential_build_and_solve():
    sp = build_solvepack_from_config(_basic_cfg(), seed=0)
    vp = solve(sp)
    assert isinstance(vp.frames, list) and len(vp.frames) >= 1
    scene = getattr(sp, "scene0", None)
    if scene is None:
        scene = sp.scene
    labels = scene.labels if hasattr(scene, "labels") else scene["labels"]
    assert len(labels) == sp.N


def test_determinism_and_contract():
    cfg = _basic_cfg()
    sp1 = build_solvepack_from_config(cfg, seed=42)
    sp2 = build_solvepack_from_config(cfg, seed=42)
    assert np.allclose(sp1.P0, sp2.P0)
    scene = getattr(sp1, "scene0", None)
    if scene is None:
        scene = sp1.scene
    WH = scene.WH if hasattr(scene, "WH") else scene["WH"]
    labels = scene.labels if hasattr(scene, "labels") else scene["labels"]
    assert WH.shape == (sp1.N, 2)
    assert len(labels) == sp1.N
    assert sp1.active0.shape == (sp1.N,)
