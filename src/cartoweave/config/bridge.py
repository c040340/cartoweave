# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import warnings


def translate_legacy_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    c = dict(cfg)
    comp = c.setdefault("compute", {})

    hits = []

    # weights
    comp.setdefault("weights", {})
    if not comp["weights"]:
        w1 = cfg.get("solver", {}).get("internals", {}).get("weights")
        w2 = cfg.get("solver", {}).get("terms", {}).get("weights")
        if w1:
            comp["weights"] = dict(w1)
            hits.append("solver.internals.weights")
        elif w2:
            comp["weights"] = dict(w2)
            hits.append("solver.terms.weights")

    # eps
    eps = comp.setdefault("eps", {})
    if "numeric" not in eps:
        legacy_eps = cfg.get("solver", {}).get("eps", {}).get("numeric")
        eps["numeric"] = 1e-12 if legacy_eps is None else legacy_eps
        if legacy_eps is not None:
            hits.append("solver.eps.numeric")

    # passes defaults (non-intrusive)
    p = comp.setdefault("passes", {})

    def _set(name, defaults):
        if name not in p:
            p[name] = dict(defaults)

    _set("grad_clip", {"max_norm": None, "max_abs": None})
    _set("nan_guard", {"e_fallback": 0.0})
    _set("capture", {"every": 1, "final_always": True})
    _set("step_limit", {"max_step_norm": None})
    _set("early_stop", {"enabled": False, "gtol": None, "ftol": None})
    _set("schedule", {})

    comp["_bridge"] = {"legacy_hits": hits, "hit_count": len(hits)}

    if hits:
        warnings.warn(
            "Using legacy config keys mapped to compute.*: " + ", ".join(hits),
            category=DeprecationWarning,
            stacklevel=2,
        )
    return c
