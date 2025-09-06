import inspect

import numpy as np

from cartoweave.data.api import build_solvepack_direct
from cartoweave.compute.forces.area_cross import evaluate as cross_eval
from cartoweave.compute.forces.area_softout import evaluate as softout_eval
from cartoweave.compute.forces.area_embed import evaluate as embed_eval
import cartoweave.compute.forces.area_cross as ac


def fd_check(eval_fn, scene, P, cfg, eps=1e-6, k=8):
    # pick k random coordinates to perturb
    E, G, _ = eval_fn(scene, P, cfg, phase="pre_anchor")
    L = P.shape[0]
    idx = np.random.default_rng(0).choice(L * 2, size=min(k, L * 2), replace=False)
    for j in idx:
        ii, d = divmod(j, 2)
        Pp = P.copy(); Pp[ii, d] += eps
        Ep, _, _ = eval_fn(scene, Pp, cfg, phase="pre_anchor")
        g_fd = (Ep - E) / eps
        assert np.isfinite(g_fd)
        assert np.isfinite(G[ii, d])
        denom = max(1.0, abs(g_fd), abs(G[ii, d]))
        assert abs(g_fd - G[ii, d]) / denom < 1e-3


def test_smoothness_smoke():
    sp = build_solvepack_direct(
        frame_size=(800, 600),
        n_labels=5,
        n_points=3,
        n_lines=2,
        n_areas=2,
        steps={"kind": "none"},
        seed=123,
    )
    scene = sp.scene
    P = sp.P0.copy()
    cfg = {"compute": {"eps": {"numeric": 1e-12}}}
    fd_check(cross_eval, scene, P, cfg)
    fd_check(softout_eval, scene, P, cfg)
    fd_check(embed_eval, scene, P, cfg)


def test_area_cross_no_intersect_token():
    assert "intersect" not in inspect.getsource(ac)
