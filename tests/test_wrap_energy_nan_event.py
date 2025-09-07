# ruff: noqa: S101
import numpy as np

from cartoweave.compute.passes import REGISTRY
from cartoweave.compute.passes.base import ComputePass
from cartoweave.compute.solve import solve
from cartoweave.data.api import make_solvepack_from_data_defaults


class InjectNaNPass(ComputePass):
    def wrap_energy(self, energy_fn):
        def _wrapped(p, labels, scene, mask, cfg):
            energy, grad, comps = energy_fn(p, labels, scene, mask, cfg)
            grad_mod = grad.copy()
            grad_mod[0, 0] = np.nan
            return energy, grad_mod, comps
        return _wrapped


def test_nan_injection_emits_event():
    REGISTRY["inject_nan"] = (InjectNaNPass, {})
    sp = make_solvepack_from_data_defaults(
        compute_cfg={"passes": {"capture": {"every": 1}}}
    )
    object.__setattr__(sp, "passes", ["inject_nan"])
    vp = solve(sp)
    events = [
        e
        for fr in vp.frames
        for e in fr.meta["events"]
        if e.get("pass") == "wrap_energy" and e.get("info") == "nonfinite"
    ]
    assert events and events[0]["metrics"].get("nonfinite_G", 0) > 0
    del REGISTRY["inject_nan"]
