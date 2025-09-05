import numpy as np
import numpy as np
from cartoweave.compute.eval import energy_and_grad_full


def random_poly(rng, n=5, cx=400.0, cy=300.0, rmin=50.0, rmax=200.0):
    ang = np.sort(rng.uniform(0.0, 2*np.pi, n))
    rad = rng.uniform(rmin, rmax, n)
    return np.stack([cx + rad*np.cos(ang), cy + rad*np.sin(ang)], axis=1)


def test_area_embed_smoke():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        poly = random_poly(rng)
        P0 = rng.uniform([300.0, 200.0], [500.0, 400.0], size=(1,2))
        WH = rng.uniform([40.0, 20.0], [120.0, 80.0], size=(1,2))
        scene = dict(
            frame=0, frame_size=(800,600),
            labels_init=P0.copy(),
            WH=WH.copy(),
            labels=[{"anchor_kind":"area","anchor_index":0}],
            areas=[{"polygon": poly}],
            points=np.zeros((0,2)), lines=np.zeros((0,2,2)),
            anchors=np.zeros((1,2)),
        )
        cfg = {"terms": {"area_embed": {"k": 200.0, "sigma": 6.0}}}
        E, G, comps, _ = energy_and_grad_full(P0.copy(), scene, np.ones(len(P0), bool), cfg)
        assert np.isfinite(E)
        assert np.isfinite(G).all()
        assert np.linalg.norm(G) < 1e12

        cfg_small = {"terms": {"area_embed": {"k": 200.0, "sigma": 1.0e-3}}}
        E2, G2, comps2, _ = energy_and_grad_full(P0.copy(), scene, np.ones(len(P0), bool), cfg_small)
        assert np.isfinite(E2)
        assert np.isfinite(G2).all()

