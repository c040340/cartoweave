from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.compute.solve import solve


def test_optimizer_step_algo_matches_solver_mode():
    compute_cfg = {
        "solver": {
            "public": {"mode": "lbfgsb", "use_warmup": False},
            "tuning": {"lbfgsb": {"lbfgs_maxiter": 2}},
        }
    }
    pack = make_solvepack_from_data_defaults(compute_cfg=compute_cfg)
    vp = solve(pack)
    algos = [
        ev.get("algo")
        for fr in vp.frames
        for ev in fr.meta.get("events", [])
        if ev.get("kind") == "optimizer_step"
    ]
    assert algos, "no optimizer_step events"
    assert all(a == "lbfgsb" for a in algos)
