from __future__ import annotations
import numpy as np
from typing import Dict, Any

def check_force_grad_consistency(Fsum: np.ndarray, g2d: np.ndarray) -> float:
    if Fsum.size == 0: return 0.0
    resid = Fsum + g2d  # F = -∇E → Fsum + g ≈ 0
    if not np.all(np.isfinite(resid)): return float("inf")
    return float(np.abs(resid).max())

def merge_sources(dst: Dict[str, Any], src: Dict[str, Any] | None) -> None:
    if not isinstance(src, dict): return
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
        else:
            if isinstance(dst[k], list) and isinstance(v, list):
                dst[k].extend(v)
            else:
                dst[k] = v
