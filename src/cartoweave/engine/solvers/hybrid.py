from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from ..core_eval import energy_and_grad_fullP
from .lbfgs import solve_layout_lbfgs
from .semi_newton import solve_layout_semi_newton

Array = np.ndarray


def _cfg(cfg: Dict[str, Any], key: str, default):
    """Safe access that avoids triggering StrictConfig for missing keys."""
    return cfg[key] if key in cfg else default


def _grad_inf(scene: Dict[str, Any], P: Array, cfg: Dict[str, Any]) -> float:
    """Infinity norm of gradient restricted to movable indices."""
    _, G, _ = energy_and_grad_fullP(scene, P, cfg)
    mov = scene.get("movable_idx")
    if mov is None:
        mov = np.arange(P.shape[0], dtype=int)
    g = G[mov].reshape(-1)
    return float(np.linalg.norm(g, np.inf))


def solve_layout_hybrid(scene, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Two-stage solver combining L-BFGS-B and Semi-Newton.

    The first stage is chosen via ``cfg['hybrid_first']`` (default ``'lbfgs'``).
    If the projected gradient norm after the first stage is above ``gtol``, the
    alternate solver is invoked.  Optionally another L-BFGS-B "polish" step can
    be performed after Semi-Newton via ``hybrid_polish_lbfgs``.
    """
    if "hybrid_gtol" in cfg:
        gtol = float(cfg["hybrid_gtol"])
    elif "sn_gtol" in cfg:
        gtol = float(cfg["sn_gtol"])
    else:
        gtol = 1e-3
    first = str(_cfg(cfg, "hybrid_first", "lbfgs")).lower()
    polish = bool(_cfg(cfg, "hybrid_polish_lbfgs", True))

    stages = []

    if first == "lbfgs":
        P1, info1 = solve_layout_lbfgs(scene, cfg)
        stages.append(("lbfgs", info1))
        if _grad_inf(scene, P1, cfg) <= gtol:
            return P1, {"stages": stages, "success": True}
        sc = dict(scene)
        sc["labels_init"] = P1
        P2, info2 = solve_layout_semi_newton(sc, cfg)
        stages.append(("semi", info2))
        P_cur, info_cur = P2, info2
        if _grad_inf(sc, P_cur, cfg) <= gtol:
            if polish:
                sc2 = dict(sc)
                sc2["labels_init"] = P_cur
                P3, info3 = solve_layout_lbfgs(sc2, cfg)
                stages.append(("lbfgs_polish", info3))
                P_cur = P3
        success = _grad_inf(sc, P_cur, cfg) <= gtol
        return P_cur, {"stages": stages, "success": success}
    else:
        P1, info1 = solve_layout_semi_newton(scene, cfg)
        stages.append(("semi", info1))
        if _grad_inf(scene, P1, cfg) <= gtol:
            return P1, {"stages": stages, "success": True}
        sc = dict(scene)
        sc["labels_init"] = P1
        P2, info2 = solve_layout_lbfgs(sc, cfg)
        stages.append(("lbfgs", info2))
        success = _grad_inf(sc, P2, cfg) <= gtol
        return P2, {"stages": stages, "success": success}
