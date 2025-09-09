import numpy as np
from cartoweave.compute.forces.area_embed import evaluate as eval_embed
from cartoweave.compute.forces.area_cross import evaluate as eval_cross
from cartoweave.compute.forces.area_softout import evaluate as eval_softout


def _scene():
    return {
        "labels": [{"WH": [2, 2], "anchor": {"target": "area", "index": 0}}],
        "areas": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
    }


def test_embed_force_applies_for_target_anchor():
    scene = _scene()
    P = np.array([[15.0, 5.0]], float)
    E, F, _ = eval_embed(scene, P, {}, {})
    assert np.linalg.norm(F) > 1e-6


def test_cross_and_softout_ignore_self_area():
    scene = _scene()
    P = np.array([[15.0, 5.0]], float)
    E_cross, F_cross, _ = eval_cross(scene, P, {}, {})
    E_soft, F_soft, _ = eval_softout(scene, P, {}, {})
    assert np.allclose(F_cross, 0.0) and np.isclose(E_cross, 0.0)
    assert np.allclose(F_soft, 0.0) and np.isclose(E_soft, 0.0)
