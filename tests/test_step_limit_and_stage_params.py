from cartoweave.contracts.solvepack import SolvePack
from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve
import numpy as np


def test_stage_params_and_step_limit_stats():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 3, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {
            "weights": {"anchor.spring": 1.0},
            "eps": {"numeric": 1e-12},
            "passes": {
                "step_limit": {"max_step_norm": 0.5},
                "capture": {"every": 1},
            },
        },
        "behaviors": [{"solver": "lbfgs", "iters": 2}],
    }
    sp = build_solvepack_from_config(cfg, seed=0)
    vp = solve(sp)
    ps = vp.summary.get("pass_stats", {}).get("StepLimitPass", {})
    assert "clamped_steps" in ps
    assert vp.summary.get("mask_popcount")
