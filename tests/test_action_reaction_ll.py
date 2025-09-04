# tests/test_action_reaction_ll.py
import numpy as np
import pytest
from cartoweave.engine.core_eval import energy_and_grad_fullP

@pytest.mark.parametrize("geom", ["rect", "disk"])
def test_ll_global_force_zero(geom):
    rng = np.random.default_rng(0)
    P0 = rng.uniform(100, 300, size=(4,2)).astype(float)
    WH = np.array([[60.,24.]]*4, float)

    scene = dict(
        frame=0, frame_size=(1000,1000),
        labels_init=P0, WH=WH,
        labels=[{"anchor_kind":"none"}]*4,
        points=np.zeros((0,2)), lines=np.zeros((0,2,2)),
        areas=np.zeros((0,6)), anchors=np.zeros((4,2)),
    )
    cfg = {"ll.geom": geom, "ll.k.repulse": 200.0, "ll.k.inside": 50.0}

    E, G, _ = energy_and_grad_fullP(scene, P0, cfg)
    # 只开 ll.* 时，总外力为 0 → ΣF = 0 → Σ(-G) = 0
    resid = float(np.abs(G.sum(axis=0)).max())
    assert resid < 1e-8, f"ll.{geom} action-reaction broken, resid={resid}"
