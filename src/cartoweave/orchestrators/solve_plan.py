from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from cartoweave.engine.solvers import lbfgs
from cartoweave.utils.logging import logger

Stage = Dict[str, Any]


def _apply_stage_cfg(base: Dict[str, Any], stage: Stage) -> Dict[str, Any]:
    """Return ``base`` updated by stage-level ``scale`` and ``override`` rules."""

    cfg = deepcopy(base)
    for path, scale in stage.get("scale", {}).items():
        if path in cfg and isinstance(cfg[path], (int, float)):
            cfg[path] = float(cfg[path]) * float(scale)
            continue
        keys = path.split(".")
        d = cfg
        for k in keys[:-1]:
            if not isinstance(d, dict) or k not in d:
                d = None
                break
            d = d[k]
        if d is not None:
            last = keys[-1]
            val = d.get(last)
            if isinstance(val, (int, float)):
                d[last] = float(val) * float(scale)

    for path, value in stage.get("override", {}).items():
        if path in cfg:
            cfg[path] = value
            continue
        keys = path.split(".")
        d = cfg
        for k in keys[:-1]:
            node = d.get(k)
            if not isinstance(node, dict):
                node = {}
                d[k] = node
            d = node
        d[keys[-1]] = value
    return cfg


def _debug_enabled(cfg: Dict[str, Any]) -> bool:
    env_flag = os.environ.get("CFG_DEBUG_FORCES", "")
    if env_flag and env_flag not in {"0", "false", "False"}:
        return True
    return bool(
        cfg.get("solver", {})
        .get("tuning", {})
        .get("debug", {})
        .get("force_stats")
    )


def _log_config_stats(cfg: Dict[str, Any]) -> None:
    tuning = cfg.get("solver", {}).get("tuning", {})
    terms = tuning.get("terms", {})
    terms_sum = float(sum(float(v.get("weight", 0.0)) for v in terms.values()))
    anchor_k = float(tuning.get("anchor", {}).get("k_spring", 0.0))
    thr = tuning.get("threshold", {})
    abs_thr = thr.get("abs")
    rel_thr = thr.get("rel")
    msg = (
        f"[cfg] terms_weight_sum={terms_sum}, anchor.k_spring={anchor_k}, "
        f"threshold.abs={abs_thr}, threshold.rel={rel_thr}"
    )
    if terms_sum <= 1e-12 and anchor_k <= 1e-12:
        msg += " all forces disabled"
    logger.info(msg)


def run_solve_plan(
    scene: Dict[str, Any],
    cfg: Dict[str, Any],
    plan: List[Stage],
    *,
    mode: str | None = None,
    carry_P: bool = True,
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None]
    | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute a simple multi-stage solve plan."""

    cfg_mode = (
        cfg.get("solver", {}).get("public", {}).get("mode", "lbfgsb")
        if isinstance(cfg, dict)
        else "lbfgsb"
    )
    mode = str(mode or cfg_mode).lower()
    profile = cfg.get("solver", {}).get("public", {}).get("profile") if isinstance(cfg, dict) else None

    if mode in {"lbfgs", "lbfgsb"}:
        solver_fn = lbfgs.run
    elif mode == "simple":  # pragma: no cover - simple solver rarely used
        solver_fn = lbfgs.run
    elif mode in {"hybrid", "newton"}:
        logger.info(
            "[orchestrator] mode=%s → fallback=lbfgsb (not implemented yet)",
            mode.upper(),
        )
        solver_fn = lbfgs.run
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if profile:
        logger.info("[orchestrator] profile preset=%s", profile)

    if not plan:
        raise ValueError("run_solve_plan received empty solve_plan")

    stages = list(plan)
    use_warmup = False
    step_cap = None
    if isinstance(cfg, dict):
        use_warmup = (
            cfg.get("solver", {}).get("public", {}).get("use_warmup", False)
        )
        step_cap = (
            cfg.get("solver", {})
            .get("tuning", {})
            .get("warmup", {})
            .get("step_cap_px")
        )
    if use_warmup and len(stages) == 1:
        warm = {
            "name": "warmup_no_anchor",
            "scale": {
                "solver.tuning.anchor.k_spring": 0.0,
                "anchor.k.spring": 0.0,
            },
        }
        if step_cap is not None:
            warm.setdefault("override", {})[
                "solver.tuning.anti_jump.step_cap_px"
            ] = float(step_cap)
        stages.insert(0, warm)
        logger.info(
            "[orchestrator] injected warmup_no_anchor (k_spring→0, step_cap_px=%s)",
            step_cap,
        )
    plan = stages

    sc = dict(scene)
    P_cur = np.asarray(sc.get("labels_init", np.zeros((0, 2), float)), float)
    history_pos: List[np.ndarray] = []
    history_E: List[float] = []
    history_rec: List[Dict[str, Any]] = []
    stages_meta: List[Dict[str, Any]] = []

    if _debug_enabled(cfg):
        _log_config_stats(cfg)

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

        info = solver_fn(sc, sc.get("labels_init", np.zeros_like(P_cur)), cfg_stage, record=_rec)
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

