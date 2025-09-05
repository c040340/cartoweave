from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from .types import SceneData
from .engine.solvers.lbfgs import solve_layout_lbfgs
from .engine.solvers.semi_newton import solve_layout_semi_newton
from .engine.solvers.hybrid import solve_layout_hybrid
from .utils.logging import logger
from .orchestrators.scene_script_runner import run_scene_script

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
      - lines:  (Nl,2,2) float polylines
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

def solve_plan(
    stages: list[Dict[str, Any]],
    scene: SceneData,
    cfg: Dict[str, Any] | None = None,
    mode: str = "hybrid",
    cache_path: str | None = None,
):
    """Run a sequence of actions (timeline). Returns (history, last_result).
    This is a placeholder orchestrator-level API; see orchestrators.solve_plan for details.
    """
    from .orchestrators.solve_plan import run_solve_plan

    logger.info("solve_plan start mode=%s", mode)
    result = run_solve_plan(stages, scene, cfg or {}, mode=mode, cache_path=cache_path)
    logger.info("solve_plan done mode=%s", mode)
    return result


def solve_scene_script(
    scene: SceneData,
    scene_script: Dict[str, Any],
    cfg: Dict[str, Any] | None = None,
    solve_plan: Dict[str, Any] | list[Dict[str, Any]] | None = None,
):
    """Apply a scene script while executing a solve plan per step."""

    if not isinstance(scene_script, dict):
        raise TypeError("scene_script must be a dict with a 'steps' list")
    steps = scene_script.get("steps")
    if not isinstance(steps, list):
        raise TypeError("scene_script['steps'] must be a list")

    plan = solve_plan
    if plan is None:
        plan = {"stages": [{"name": "main"}]}

    cfg = cfg or {}
    return run_scene_script(scene, steps, plan, cfg)
