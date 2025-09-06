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


from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def test_integration_schedule_and_capture():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 3, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {"passes": {"capture": {"every": 1}, "step_limit": {"max_step_norm": 1.5}}},
        "behaviors": [{"solver": "lbfgs", "iters": 2}],
    }
    sp = build_solvepack_from_config(cfg, seed=0)
    view = solve(sp)
    assert view.summary.get("frames_captured", 0) >= 1
    assert "pass_stats" in view.summary
