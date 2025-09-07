# -*- coding: utf-8 -*-
"""Compute-side force registry and enablement.

Each force term lives in ``compute/forces/<name>.py`` and registers an
``evaluate(scene, P, params, cfg) -> (E, F, meta)`` where ``F`` is ``(L,2)``.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple
import numpy as np

REGISTRY: Dict[str, Callable[[dict, np.ndarray, dict, dict], Tuple[float, np.ndarray, dict]]] = {}

def register(name: str):
    """Decorator registering a compute-side force implementation."""
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco

def term_params_map(compute_cfg: dict) -> Dict[str, dict]:
    public = (compute_cfg.get("public") or {})
    forces = (public.get("forces") or {})
    out: Dict[str, dict] = {}
    for name, params in forces.items():
        if not isinstance(params, dict):
            continue
        out[str(name)] = params
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
from .anchors import evaluate as _anchors_eval  # anchor.spring
from .boundary import evaluate as _boundary_eval  # boundary.wall
from .label_label_disk import evaluate as _lldisk_eval  # ll.disk
from .focus import evaluate as _focus_eval  # focus.attract
from .point_label import evaluate as _pl_eval  # pl.rect
from .line_label import evaluate as _lline_eval  # ln.rect
from .area_embed import evaluate as _aembed_eval  # area.embed
from .area_cross import evaluate as _across_eval  # area.cross
from .area_softout import evaluate as _asoft_eval  # area.softout
from .label_label_rect import evaluate as _llrect_eval  # ll.rect

__all__ = ["REGISTRY", "register", "term_params_map", "enabled_terms"]
