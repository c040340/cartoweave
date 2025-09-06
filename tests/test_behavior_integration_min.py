import numpy as np
from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def test_behavior_repeated_runs_stable():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 2, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }
    sp = build_solvepack_from_config(cfg, seed=0)
    vp1 = solve(sp)
    vp2 = solve(sp)
    np.testing.assert_allclose(vp1.last.P, vp2.last.P, atol=1e-9)

