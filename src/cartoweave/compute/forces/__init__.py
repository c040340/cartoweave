# -*- coding: utf-8 -*-
"""Compute-side force registry and enablement.

Each force term lives in ``compute/forces/<name>.py`` and registers an
``evaluate(scene, P, cfg, phase) -> (E, F, meta)`` where ``F`` is (L,2).
"""

from __future__ import annotations
from typing import Dict, Callable, Tuple, Any, List
import numpy as np

REGISTRY: Dict[str, Callable[[dict, np.ndarray, dict, str], Tuple[float, np.ndarray, dict]]] = {}

def register(name: str):
    """Decorator registering a compute-side force implementation."""
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco

def enabled_terms(cfg: dict, *, phase: str) -> List[str]:
    """Return enabled term names for a given phase ("pre_anchor" | "anchor").

    Read weights from the *most compatible* location to ease migration:
      - preferred: cfg["compute"]["weights"]
      - legacy:    cfg["solver"]["internals"]["weights"]
      - legacy:    cfg["solver"]["terms"]["weights"]

    Enable term if weight > 0. For phases:
      - "pre_anchor": any term NOT starting with "anchor."
      - "anchor":     only terms starting with "anchor."
    """
    ws = cfg.get("weights", {}) or {}
    names = [k for k, v in ws.items() if float(v) > 0.0]
    if phase == "pre_anchor":
        return [n for n in names if not n.startswith("anchor.")]
    if phase == "anchor":
        return [n for n in names if n.startswith("anchor.")]
    return []

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

__all__ = ["REGISTRY", "register", "enabled_terms"]
