# -*- coding: utf-8 -*-
"""Compute-side force registry and enablement.

Each force term lives in ``compute/forces/<name>.py`` and registers an
``evaluate(scene, P, params, cfg) -> (E, F, meta)`` where ``F`` is ``(L,2)``.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np

REGISTRY: Dict[str, Callable] = {}
PROBES: Dict[str, Callable] = {}


def register(name: str, *, probe: Optional[Callable] = None):
    """Decorator registering a compute-side force implementation."""

    def deco(fn):
        REGISTRY[name] = fn
        if probe is not None:
            PROBES[name] = probe
        return fn

    return deco


def register_probe(name: str):
    """Decorator registering only a probe implementation for ``name``."""

    def deco(fn):
        PROBES[name] = fn
        return fn

    return deco


def get_evaluate(term: str) -> Optional[Callable]:
    return REGISTRY.get(term)


def get_probe(term: str) -> Optional[Callable]:
    """Return the registered ``probe`` callable for ``term`` if any."""

    return PROBES.get(term)


def _sub(cfg: dict, *path, default=None):
    """Shallow helper to traverse ``cfg`` with ``path``."""
    d = cfg
    for k in path:
        d = (d or {}).get(k, {})
    if d:
        return d
    return default if default is not None else {}


def term_cfg(cfg: dict, *path) -> dict:
    """Return term configuration under ``compute.public.forces``."""
    comp = (cfg or {}).get("compute", cfg or {})
    return _sub(comp, "public", "forces", *path)


def kernel_params(tc: dict, defaults: dict) -> dict:
    ker = (tc.get("kernel") or {})
    def _g(key: str, default):
        val = ker.get(key)
        return default if val is None else val
    return {
        "kernel_model": _g("model", defaults.get("model", "inv_pow")),
        "kernel_exponent": float(_g("exponent", defaults.get("exponent", 2.0))),
        "kernel_soft_eps": float(_g("soft_eps", defaults.get("soft_eps", 1e-6))),
        "kernel_gate_gamma": float(_g("gate_gamma", defaults.get("gate_gamma", 8.0))),
    }


def eps_params(cfg: dict, tc: dict, defaults: dict) -> dict:
    # accept both full root config and compute-only config
    comp_eps = _sub(cfg, "compute", "eps")
    if not comp_eps:
        comp_eps = _sub(cfg, "eps")
    tc_eps = (tc.get("eps") or {})

    def _g(key: str, fallback: float) -> float:
        if key in tc_eps and tc_eps[key] is not None:
            return float(tc_eps[key])
        if key in comp_eps:
            return float(comp_eps[key])
        if defaults is not None and key in defaults:
            return float(defaults[key])
        return float(fallback)

    return {
        "eps_numeric": _g("numeric", 1e-12),
        "eps_dist": _g("dist", 1e-6),
        "eps_abs": _g("abs", 1e-3),
        "eps_proj": _g("proj", 1e-9),
    }

def term_params_map(compute_cfg: dict) -> Dict[str, dict]:
    public = (compute_cfg.get("public") or {})
    forces = (public.get("forces") or {})
    out: Dict[str, dict] = {}
    for grp, gparams in forces.items():
        if not isinstance(gparams, dict):
            continue
        for term, params in gparams.items():
            if isinstance(params, dict):
                out[f"{grp}.{term}"] = params
    return out


def enabled_terms(compute_cfg: dict, *, phase: str) -> List[str]:
    """Return a list of enabled term names filtered by phase."""
    pmap = term_params_map(compute_cfg)
    names = [name for name, p in pmap.items() if bool(p.get("enable", False))]
    if phase == "pre_anchor":
        return [n for n in names if not n.startswith("anchor.")]
    if phase == "anchor":
        return [n for n in names if n.startswith("anchor.")]
    return names

# Keep existing explicit imports of migrated/wrapper terms below.
# Example (do NOT remove your existing ones if present):
# from .anchors import evaluate as _anchors_eval
# from .boundary import evaluate as _boundary_eval
# ...
from .anchor_spring import evaluate as _anchors_eval  # anchor.spring
from .boundary import evaluate as _boundary_eval  # boundary.wall
from .label_label_disk import evaluate as _lldisk_eval  # ll.disk
from .focus import evaluate as _focus_eval  # focus.attract
from .point_label import evaluate as _pl_eval  # pl.rect
from .line_label import evaluate as _lline_eval  # ln.rect
from .area_embed import evaluate as _aembed_eval  # area.embed
from .area_cross import evaluate as _across_eval  # area.cross
from .area_softout import evaluate as _asoft_eval  # area.softout
from .label_label_rect import evaluate as _llrect_eval  # ll.rect

__all__ = [
    "REGISTRY",
    "register",
    "register_probe",
    "get_evaluate",
    "get_probe",
    "term_params_map",
    "enabled_terms",
]
