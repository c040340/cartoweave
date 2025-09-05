# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np


def get_eps(cfg: Dict[str, Any]) -> float:
    """Retrieve numeric epsilon from configuration.

    Prefers compute.eps.numeric, then solver.eps.numeric, and
    finally falls back to machine epsilon.
    """
    return (
        cfg.get("compute", {}).get("eps", {}).get("numeric")
        or cfg.get("solver", {}).get("eps", {}).get("numeric")
        or np.finfo(float).eps
    )


def weight_of(term: str, cfg: Dict[str, Any], default: float = 0.0) -> float:
    """Lookup force weight for *term* from configuration."""
    ws = (
        cfg.get("compute", {}).get("weights", {})
        or cfg.get("solver", {}).get("internals", {}).get("weights", {})
        or cfg.get("solver", {}).get("terms", {}).get("weights", {})
        or {}
    )
    return float(ws.get(term, default))


def ensure_vec2(F: np.ndarray, L: int) -> np.ndarray:
    """Validate that *F* is an ``(L,2)`` array of floats."""
    F = np.asarray(F, float)
    if F.shape != (L, 2):
        raise ValueError(f"[TERM SHAPE MISMATCH] F={F.shape}, expected {(L, 2)}")
    return F
