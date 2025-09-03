from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from .semi_newton import solve_layout_semi_newton
from .lbfgs import solve_layout_lbfgs

def solve_layout_hybrid(scene, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    P1, info1 = solve_layout_semi_newton(scene, cfg)
    scene2 = dict(scene)
    scene2["labels_init"] = P1
    P2, info2 = solve_layout_lbfgs(scene2, cfg)
    info = {"stage1": info1, "stage2": info2}
    return P2, info
