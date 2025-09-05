from cartoweave.compute import SolvePack, solve
from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.compute.viewpack_utils import to_old_payload
import numpy as np

def test_viewpack_and_mapping(P0, mask, scene, L):
    cfg = {"compute": {"passes": {"capture": {"every": 1}}}}
    sp = SolvePack(L=L, P0=P0, active_mask0=mask, scene=scene,
                   params={"max_iter":3, "terms":{"anchor":{"spring":{"k":5.0}}}},
                   energy_and_grad=energy_and_grad_full, cfg=cfg)
    sp.passes=["schedule", "capture"]
    vp = solve(sp)
    assert vp.frames and vp.last and "E_last" in vp.summary
    old = to_old_payload(vp)
    for k in ("P_seq","E_seq","G_seq","comps_seq","mask_seq","stage_seq","summary"):
        assert k in old
