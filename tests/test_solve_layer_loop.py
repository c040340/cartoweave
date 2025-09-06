import numpy as np
from cartoweave.compute.solver.solve_layer import run_iters, SolveContext


def quad_energy(P, labels, scene, active, cfg):
    # E = 0.5 * ||P||^2, g = P
    E = 0.5 * float(np.sum(P * P))
    return E, P.copy(), {}


def test_converges_under_tols():
    P0 = np.ones((4, 2), float)
    ctx = SolveContext(labels=None, scene=None,
                       active=np.array([True, True, True, True]),
                       cfg={"solver": {"gtol": 1e-9, "ftol": 1e-12, "xtol": 1e-12}},
                       iters=200, mode="lbfgs", params={"step": 0.5})
    Pf, reps = run_iters(P0, ctx, quad_energy, report=True)
    assert np.max(np.abs(Pf)) < 1e-4


def test_clip_limits_update():
    P0 = np.ones((2, 2), float) * 100.0
    ctx = SolveContext(labels=None, scene=None,
                       active=np.array([True, True]),
                       cfg={"solver": {"gtol": 1e-9}},
                       iters=1, mode="lbfgs", params={"step": 1.0, "max_step_norm": 0.1})
    Pf, _ = run_iters(P0, ctx, quad_energy)
    # One clipped step, so Pf should be closer to P0 but not zero
    assert np.max(np.abs(Pf)) < np.max(np.abs(P0))
    assert np.max(np.abs(Pf)) > 0.0
