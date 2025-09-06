import numpy as np
from cartoweave.compute.eval import energy_and_grad_full as eg
import pytest

def test_compute_aggregator_invariants(rng, L, scene, mask):
    P = rng.standard_normal((L, 2)).astype(float)
    cfg = {
        "terms": {"anchor": {"spring": {"k": 10.0}}, "ll": {"disk": {"k": 1.0}}},
        "compute": {"weights": {"anchor.spring": 1.0, "ll.disk": 1.0}},
    }
    labels = scene["labels"]
    E, G, comps = eg(P, labels, scene, mask, cfg)
    assert comps, "comps must exist"
    G_from = -sum(comps.values())
    assert np.allclose(G, G_from, atol=1e-6)
    assert np.all(G[~mask] == 0.0)
    for v in comps.values():
        assert np.all(v[~mask] == 0.0)
