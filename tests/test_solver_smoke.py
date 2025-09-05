import numpy as np


def test_lbfgs_quadratic_smoke():
    from cartoweave.compute.optim import run_solver

    rng = np.random.RandomState(0)
    L = 16
    A = np.eye(2 * L)
    b = rng.randn(2 * L)

    def f(P):
        x = P.reshape(-1)
        return 0.5 * x.dot(A).dot(x) - b.dot(x)

    def g(P):
        x = P.reshape(-1)
        return (A.dot(x) - b).reshape((-1, 2))

    P0 = rng.randn(L, 2)
    res = run_solver("lbfgs", P0, f, g, params={"max_iter": 60})
    assert "P" in res and res["P"].shape == (L, 2)
    assert res.get("iters", 0) > 0
    assert np.isfinite(res.get("E", np.inf))


def test_integration_schedule_and_capture():
    import numpy as np
    from cartoweave.compute.pack import SolvePack
    from cartoweave.compute.run import solve

    L = 8
    P0 = np.zeros((L, 2), float)
    scene = {"labels_init": np.zeros((L, 2), float), "labels": [{"anchor_kind": "none"}] * L}
    cfg = {"compute": {"weights": {"anchor.spring": 1.0}, "eps": {"numeric": 1e-12}}}
    sp = SolvePack(
        L=L,
        P0=P0,
        active_mask0=np.ones(L, dtype=bool),
        scene=scene,
        cfg=cfg,
        stages=[{"iters": 3, "solver": "lbfgs"}, {"iters": 2, "solver": "lbfgs"}],
        passes=["schedule", "capture"],
    )
    view = solve(sp)
    assert view.summary["frames_captured"] >= 1
