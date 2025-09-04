from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, Literal, TypedDict

Mutability = Literal["frozen", "action", "runtime"]


class SpecItem(TypedDict, total=False):
    type: type                     # expected Python type
    range: Optional[Tuple[float, float]]  # numeric inclusive bounds
    choices: Optional[Tuple[Any, ...]]    # enum choices
    mutable: Mutability            # mutability class
    group: str                     # optional taxonomy: "shape", "calib_k", etc.
    default: Any                   # default (informational; not applied here)

# Minimal initial SPEC; extend later safely
SPEC: Dict[str, SpecItem] = {
    # ---- shape (frozen) ----
    "ll.edge_power":        {"type": float, "range": (1.0, 6.0), "mutable": "frozen", "group":"shape"},
    "boundary.wall_power":  {"type": float, "range": (1.0, 6.0), "mutable": "frozen", "group":"shape"},
    "beta.softplus.dist":   {"type": float, "range": (0.1, 10.0), "mutable": "frozen", "group":"shape"},
    "anchor.spring.alpha":  {"type": float, "range": (0.1, 50.0), "mutable": "frozen", "group":"shape"},
    "eps.abs":              {"type": float, "range": (1e-12, 10.0), "mutable": "frozen", "group":"shape"},

    # ---- k (action) ----
    "focus.k.attract":      {"type": float, "range": (0.0, 1e9), "mutable": "action", "group":"calib_k"},
    "ll.k.repulse":         {"type": float, "range": (0.0, 1e9), "mutable": "action", "group":"calib_k"},
    "boundary.k.wall":      {"type": float, "range": (0.0, 1e9), "mutable": "action", "group":"calib_k"},

    # ---- calibration controls (action) ----
    "calib.k.enable":       {"type": bool,  "mutable": "action", "group":"trigger"},
    "calib.k.clip_q":       {"type": float, "range": (0.5, 1.0), "mutable": "action", "group":"trigger"},
    "calib.k.p_q":          {"type": float, "range": (0.5, 0.999), "mutable": "action", "group":"trigger"},
    "calib.k.min_act":      {"type": float, "range": (0.0, 1.0), "mutable": "action", "group":"trigger"},
    "calib.k.hysteresis":   {"type": float, "range": (0.0, 0.95), "mutable": "action", "group":"trigger"},
    "calib.k.clamp_min":    {"type": float, "range": (0.0, 10.0), "mutable": "action", "group":"trigger"},
    "calib.k.clamp_max":    {"type": float, "range": (0.0, 10.0), "mutable": "action", "group":"trigger"},
    "calib.k.base_term":    {"type": str,   "mutable": "action", "group":"trigger"},

    "calib.shape.enable":   {"type": bool,  "mutable": "action", "group":"shape"},
    "calib.shape.name":     {"type": str,   "mutable": "action", "group":"shape"},
    "calib.shape.sigma_scale":{"type": float,"range": (0.01, 100.0),"mutable":"action","group":"shape"},

    "calib.trigger.enable": {"type": bool,  "mutable": "action", "group":"trigger"},
    "calib.trigger.tau_on": {"type": float, "range": (0.0, 1.0), "mutable": "action", "group":"trigger"},
    "calib.trigger.tau_off":{"type": float, "range": (0.0, 1.0), "mutable": "action", "group":"trigger"},
    "calib.trigger.delta":  {"type": float, "range": (0.0, 1.0), "mutable": "action", "group":"trigger"},

    # ---- visualization (runtime) ----
    "viz.field.aspect": {"type": tuple, "mutable": "runtime", "group": "viz"},
    "viz.field.nx": {"type": int, "mutable": "runtime", "group": "viz"},
    "viz.field.ny": {"type": int, "mutable": "runtime", "group": "viz"},

    # ---- runtime example ----
    "logging.level":        {"type": str, "choices": ("DEBUG","INFO","WARNING","ERROR"), "mutable": "runtime", "group":"logging"},
}


def spec_of(key: str) -> Optional[SpecItem]:
    return SPEC.get(key)
