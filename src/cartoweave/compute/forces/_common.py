# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np

def get_eps(cfg: Dict[str, Any]) -> float:
    # Bridge-friendly: prefer compute.eps.numeric, then fallbacks
    return (
        cfg.get("compute", {}).get("eps", {}).get("numeric")
        or cfg.get("solver", {}).get("eps", {}).get("numeric")
        or np.finfo(float).eps
    )

def weight_of(term: str, cfg: Dict[str, Any], default: float = 0.0) -> float:
    # Prefer compute.weights; fall back to legacy locations
    ws = (
        cfg.get("compute", {}).get("weights", {})
        or cfg.get("solver", {}).get("internals", {}).get("weights", {})
        or cfg.get("solver", {}).get("terms", {}).get("weights", {})
        or {}
    )
    return float(ws.get(term, default))

def ensure_vec2(F: np.ndarray, L: int) -> np.ndarray:
    F = np.asarray(F, float)
    if F.shape != (L, 2):
        raise ValueError(f"[TERM SHAPE MISMATCH] F={F.shape}, expected {(L,2)}")
    return F
