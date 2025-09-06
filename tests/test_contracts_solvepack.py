import numpy as np
import pytest
from cartoweave.contracts.solvepack import SolvePack



def _scene_ok(L=3):
    return dict(
        frame_size=(800, 600),
        labels=[{"anchor_kind": "none", "anchor_index": -1} for _ in range(L)],
        WH=np.ones((L, 2), float) * 24.0,
        points=np.zeros((0, 2), float),
        lines=[],
        areas=[],
    )


def test_missing_labels_fails():
    L = 2
    scene = _scene_ok(L)
    scene.pop("labels")
    with pytest.raises(ValueError, match="scene missing required key: 'labels'"):
        SolvePack(L=L, P0=np.zeros((L, 2)), active_mask0=np.ones(L, bool),
                  scene=scene, cfg={}, stages=[{"iters": 1}], passes=["schedule"])


def test_WH_shape_mismatch_fails():
    L = 2
    scene = _scene_ok(L)
    scene["WH"] = np.ones((L + 1, 2))
    with pytest.raises(ValueError, match=r"WH first dim must be 2"):
        SolvePack(L=L, P0=np.zeros((L, 2)), active_mask0=np.ones(L, bool),
                  scene=scene, cfg={}, stages=[{"iters": 1}], passes=["schedule"])


def test_active_mask_dtype_fails():
    L = 2
    scene = _scene_ok(L)
    with pytest.raises(ValueError, match="active_mask0 must be bool array"):
        SolvePack(L=L, P0=np.zeros((L, 2)), active_mask0=np.ones(L, dtype=int),
                  scene=scene, cfg={}, stages=[{"iters": 1}], passes=["schedule"])
