from cartoweave.contracts.solvepack import SolvePack
from cartoweave.compute import solve
from cartoweave.compute.eval import energy_and_grad_full
import numpy as np

def test_schedule_and_capture(P0, mask, scene, L):
    params={"max_iter":4, "terms":{"anchor":{"spring":{"k":5.0}}}}
    cfg={"compute": {"passes": {"capture": {"every": 1}}}}
    sp = SolvePack(
        L=L,
        P0=P0,
        active_mask0=mask,
        scene=scene,
        params=params,
        energy_and_grad=energy_and_grad_full,
        cfg=cfg,
        stages=[{"solver": "semi_newton", "iters": 2}, {"solver": "lbfgs", "iters": 2}],
        passes=["schedule", "capture"],
    )
    vp = solve(sp)
    stages = {f.stage for f in vp.frames}
    assert stages=={0,1}
    assert vp.summary["stage_solvers"]==["semi_newton","lbfgs"]
    assert vp.summary["frames_captured"]==len(vp.frames)
