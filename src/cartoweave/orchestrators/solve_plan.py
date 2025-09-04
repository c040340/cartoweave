from __future__ import annotations
from typing import Dict, Any, List, Callable
import numpy as np

from cartoweave.engine.solvers import lbfgs

Stage = Dict[str, Any]


def _apply_stage_cfg(base: Dict[str, Any], stage: Stage) -> Dict[str, Any]:
    cfg = dict(base)
    for k, s in stage.get("scale", {}).items():
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = float(cfg[k]) * float(s)
    for k, v in stage.get("override", {}).items():
        cfg[k] = v
    return cfg


def run_solve_plan(
    scene: Dict[str, Any],
    cfg: Dict[str, Any],
    plan: List[Stage],
    *,
    mode: str = "lbfgs",
    carry_P: bool = True,
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None]
    | None = None,
):
    """Execute a simple multi-stage solve plan.

    Parameters
    ----------
    scene:
        Base scene dictionary shared by all stages.
    cfg:
        Base solver/physics configuration dictionary.
    plan:
        Sequence of stage dictionaries supporting optional ``scale`` and
        ``override`` keys.
    mode:
        Solver mode.  Only ``"lbfgs"`` is implemented in this minimal runner.
    carry_P:
        If True, the result of one stage seeds the next stage's ``labels_init``.
    record:
        Optional callback receiving per-evaluation snapshots.
    """

    if mode not in {"lbfgs", "hybrid", "semi_newton"}:
        raise NotImplementedError(f"Unsupported mode: {mode}")
    if not plan:
        raise ValueError("run_solve_plan received empty solve_plan")

    sc = dict(scene)
    P_cur = np.asarray(sc.get("labels_init", np.zeros((0, 2), float)), float)
    history_pos: List[np.ndarray] = []
    history_E: List[float] = []
    history_rec: List[Dict[str, Any]] = []
    stages_meta: List[Dict[str, Any]] = []

    for stage_idx, stage in enumerate(plan):
        cfg_stage = _apply_stage_cfg(cfg, stage)
        stage_name = stage.get("name", f"stage_{stage_idx}")
        if carry_P:
            sc["labels_init"] = P_cur

        def _rec(P, E, comps, meta):
            meta = dict(meta) if meta else {}
            meta.setdefault("stage_id", stage_idx)
            meta.setdefault("stage_name", stage_name)
            if record:
                record(P, E, comps, meta)

        info = lbfgs.run(sc, sc["labels_init"], cfg_stage, record=_rec)
        P_cur = info.get("P", P_cur)
        hist = info.get("history", {})
        pos = list(hist.get("positions", []))
        eng = list(hist.get("energies", []))
        rec = list(hist.get("records", []))
        for r in rec:
            meta = r.setdefault("meta", {})
            meta.setdefault("stage_id", stage_idx)
            meta.setdefault("stage_name", stage_name)
        if history_pos:
            if pos:
                pos = pos[1:]
                eng = eng[1:]
                rec = rec[1:]
        history_pos.extend(pos)
        history_E.extend(eng)
        history_rec.extend(rec)
        stages_meta.append({"name": stage_name})

    history = {"positions": history_pos, "energies": history_E, "records": history_rec}
    info = {"solve_plan": stages_meta, "history": history}
    return P_cur, info


__all__ = ["run_solve_plan"]
