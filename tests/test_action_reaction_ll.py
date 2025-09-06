import numpy as np
import pytest
from cartoweave.compute.eval import energy_and_grad_full


@pytest.mark.parametrize("geom", ["rect", "disk"])
def test_ll_global_force_zero(geom, make_min_scene):
    P, labels, scene, active, cfg = make_min_scene(L=6, seed=0)
    cfg["compute"]["weights"] = {
        "anchor.spring": 0.0,
        "boundary.wall": 0.0,
        "pl.rect": 0.0,
        "ll.rect": 0.0,
    }
    key = "ll.rect" if geom == "rect" else "ll.disk"
    cfg["compute"]["weights"][key] = 1.0

    E, g, comps = energy_and_grad_full(P, labels, scene, active, cfg)

    Gsum = np.zeros_like(g)
    for v in comps.values():
        Gsum += v
    np.testing.assert_allclose(g, -Gsum, rtol=1e-6, atol=1e-6)

    assert np.linalg.norm(g.sum(axis=0), ord=np.inf) < 1e-6
