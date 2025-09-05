from cartoweave.compute import SolvePack, solve
from cartoweave.compute.eval import energy_and_grad_full
import numpy as np

def test_schedule_and_capture(P0, mask, scene, L):
    params={"max_iter":4, "terms":{"anchor":{"spring":{"k":5.0}}}}
    sp = SolvePack(L=L, P0=P0, active_mask0=mask, scene=scene, params=params,
                   energy_and_grad=energy_and_grad_full)
    sp.schedule=[{"solver":"semi","iters":2},{"solver":"lbfgs","iters":2}]
    sp.passes=["schedule", {"name":"capture","args":{"every":1}}]
    vp = solve(sp)
    stages = {f.stage for f in vp.frames}
    assert stages=={0,1}
    assert vp.summary["stage_solvers"]==["semi","lbfgs"]
    assert vp.summary["frames_captured"]==len(vp.frames)
