# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np


def get_eps(cfg: Dict[str, Any]) -> float:
    """Retrieve numeric epsilon from configuration.

    Prefers compute.eps.numeric and
    finally falls back to machine epsilon.
    """
    comp = (cfg or {}).get("compute", {})
    eps = (comp.get("eps") or {})
    return float(eps.get("numeric", 1.0e-12))


def ensure_vec2(F: np.ndarray, L: int) -> np.ndarray:
    """Validate that *F* is an ``(L,2)`` array of floats."""
    F = np.asarray(F, float)
    if F.shape != (L, 2):
        raise ValueError(f"[TERM SHAPE MISMATCH] F={F.shape}, expected {(L, 2)}")
    return F
