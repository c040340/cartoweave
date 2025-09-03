from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from .types import SceneData
from .engine.solvers.lbfgs import solve_layout_lbfgs
from .engine.solvers.semi_newton import solve_layout_semi_newton
from .engine.solvers.hybrid import solve_layout_hybrid
from .utils.logging import logger

def solve_frame(
    scene: SceneData,
    cfg: Dict[str, Any] | None = None,
    mode: str = "lbfgs",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Public API: input a SceneData, return (P_opt, info).
    SceneData keys:
      - frame: int
      - frame_size: tuple[int,int]  (W,H)
      - points: (Np,2) float
      - lines:  (Nl,4) float  (placeholder format)
      - areas:  (Na, >= 6) float (placeholder packed polygon)
      - labels_init: (N,2) float (optional)
    """
    cfg = cfg or {}
    logger.info("solve_frame start mode=%s", mode)
    if mode == "lbfgs":
        P, info = solve_layout_lbfgs(scene, cfg)
    elif mode == "semi_newton":
        P, info = solve_layout_semi_newton(scene, cfg)
    elif mode == "hybrid":
        P, info = solve_layout_hybrid(scene, cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    msg = info.get("msg", info.get("message", ""))
    nit = info.get("nit", "?")
    logger.info("solve_frame done mode=%s nit=%s %s", mode, nit, msg)
    return P, info

def solve_timeline(
    actions: list[Dict[str, Any]],
    scene: SceneData,
    cfg: Dict[str, Any] | None = None,
    mode: str = "hybrid",
    cache_path: str | None = None,
):
    """Run a sequence of actions (timeline). Returns (history, last_result).
    This is a placeholder orchestrator-level API; see orchestrators.timeline for details.
    """
    from .orchestrators.timeline import run_timeline

    logger.info("solve_timeline start mode=%s", mode)
    result = run_timeline(actions, scene, cfg or {}, mode=mode, cache_path=cache_path)
    logger.info("solve_timeline done mode=%s", mode)
    return result
