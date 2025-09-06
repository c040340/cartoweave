from cartoweave.contracts.solvepack import SolvePack
from cartoweave.compute import solve
from cartoweave.compute.eval import energy_and_grad_full
import numpy as np


def test_stage_params_and_step_limit_stats(P0, mask, scene, L):
    cfg = {
        "compute": {
            "weights": {"anchor.spring": 1.0},
            "eps": {"numeric": 1e-12},
            "passes": {
                "step_limit": {"max_step_norm": 0.5},
                "capture": {"every": 1},
            },
        }
    }
    sp = SolvePack(
        L=L,
        P0=P0,
        active_mask0=mask,
        scene=scene,
        params={"max_iter": 2},
        energy_and_grad=energy_and_grad_full,
        cfg=cfg,
        stages=[{"solver": "lbfgs", "iters": 2}],
        passes=["schedule", "step_limit", "capture"],
    )
    vp = solve(sp)
    ps = vp.summary.get("pass_stats", {}).get("StepLimitPass", {})
    assert "max_observed_step_norm" in ps
    assert vp.summary["stages_params"]
    assert vp.summary["mask_popcount"][0] == int(mask.sum())
