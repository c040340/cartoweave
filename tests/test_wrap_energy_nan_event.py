import numpy as np
from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.passes import REGISTRY
from cartoweave.compute.passes.base import ComputePass
from cartoweave.compute.run import solve


class InjectNaNPass(ComputePass):
    def wrap_energy(self, energy_fn):
        def _wrapped(P, labels, scene, mask, cfg):
            E, G, comps = energy_fn(P, labels, scene, mask, cfg)
            G = G.copy()
            G[0, 0] = np.nan
            return E, G, comps
        return _wrapped


def test_nan_injection_emits_event():
    REGISTRY["inject_nan"] = (InjectNaNPass, {})
    cfg = {
        "data": {"generate": {"num_labels": 1}},
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }
    sp = build_solvepack_from_config(cfg, seed=1)
    object.__setattr__(sp, "passes", ["inject_nan"])
    vp = solve(sp)
    events = [e for e in vp.events if e.get("pass") == "wrap_energy" and e.get("info") == "nonfinite"]
    assert events and events[0]["metrics"].get("nonfinite_G", 0) > 0
    del REGISTRY["inject_nan"]
