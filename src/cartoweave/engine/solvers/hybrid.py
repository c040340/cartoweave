from __future__ import annotations
from typing import Dict, Any, Tuple, Callable
import numpy as np

from .lbfgs import solve_layout_lbfgs
from .semi_newton import solve_layout_semi_newton
from ...utils.logging import logger

Array = np.ndarray


def _cfg(cfg: Dict[str, Any], key: str, default):
    """Safe access that avoids triggering StrictConfig for missing keys."""
    return cfg[key] if key in cfg else default


def _grad_inf(scene: Dict[str, Any], P: Array, cfg: Dict[str, Any], energy_fn) -> float:
    """Infinity norm of gradient restricted to movable indices."""
    _, G = energy_fn(scene, P, cfg)
    mov = scene.get("movable_idx")
    if mov is None:
        mov = np.arange(P.shape[0], dtype=int)
    g = G[mov].reshape(-1)
    return float(np.linalg.norm(g, np.inf))


def solve_layout_hybrid(
    scene,
    cfg: Dict[str, Any],
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None]
    | None = None,
    energy_fn: Callable | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    logger.info("Hybrid solver start first=%s polish=%s", first, polish)

    stages = []
    history_pos: list[np.ndarray] = []
    history_E: list[float] = []
    history_rec: list[Dict[str, Any]] = []

    current_stage = ""

    def _rec_stage(P, E, comps, meta):
        m = {"stage": current_stage}
        if meta:
            m.update(meta)
        si = dict(m.get("solver_info", {}))
        si["solver"] = "hybrid"
        si["stage"] = current_stage
        si["gtol"] = gtol
        m["solver_info"] = si
        if record:
            record(P, E, comps, m)

    def _extend_history(info: Dict[str, Any], skip_first: bool = False) -> None:
        hist = info.get("history") if isinstance(info, dict) else None
        if not isinstance(hist, dict):
            return
        pos = list(hist.get("positions", []))
        eng = list(hist.get("energies", []))
        rec = list(hist.get("records", []))
        if skip_first and pos:
            pos = pos[1:]
            eng = eng[1:]
            rec = rec[1:] if rec else []
        history_pos.extend(pos)
        history_E.extend(eng)
        history_rec.extend(rec)

    def _annotate(info: Dict[str, Any], stage_name: str) -> None:
        for r in info.get("history", {}).get("records", []):
            meta = r.setdefault("meta", {})
            si = dict(meta.get("solver_info", {}))
            si["solver"] = "hybrid"
            si["stage"] = stage_name
            si["gtol"] = gtol
            meta["solver_info"] = si
            meta.setdefault("stage", stage_name)

    if energy_fn is None:
        from ...compute.eval import energy_and_grad_full as _eval

        def energy_fn(sc, P, cfg_, record=None):
            mask = np.ones(P.shape[0], bool)
            E, G, comps, _ = _eval(P, sc, mask, cfg_)
            if record is not None:
                record(P, E, comps, {})
            return E, G

    if first == "lbfgs":
        logger.info("Hybrid stage: L-BFGS")
        current_stage = "lbfgs"
        P1, info1 = solve_layout_lbfgs(scene, cfg, record=_rec_stage, energy_fn=energy_fn)
        _annotate(info1, "lbfgs")
        stages.append(("lbfgs", info1))
        _extend_history(info1, skip_first=False)
        if _grad_inf(scene, P1, cfg, energy_fn) <= gtol:
            logger.info("Hybrid solver finished after L-BFGS stage")
            return P1, {"stages": stages, "success": True, "history": {"positions": history_pos, "energies": history_E, "records": history_rec}}
        sc = dict(scene)
        sc["labels_init"] = P1
        logger.info("Hybrid stage: Semi-Newton")
        current_stage = "semi"
        P2, info2 = solve_layout_semi_newton(sc, cfg, record=_rec_stage, energy_fn=energy_fn)
        _annotate(info2, "semi")
        stages.append(("semi", info2))
        _extend_history(info2, skip_first=True)
        P_cur = P2
        if _grad_inf(sc, P_cur, cfg, energy_fn) <= gtol:
            if polish:
                sc2 = dict(sc)
                sc2["labels_init"] = P_cur
                logger.info("Hybrid stage: L-BFGS polish")
                current_stage = "lbfgs_polish"
                P3, info3 = solve_layout_lbfgs(sc2, cfg, record=_rec_stage, energy_fn=energy_fn)
                _annotate(info3, "lbfgs_polish")
                stages.append(("lbfgs_polish", info3))
                _extend_history(info3, skip_first=True)
                P_cur = P3
            success = _grad_inf(sc, P_cur, cfg) <= gtol
            logger.info("Hybrid solver finished success=%s", success)
            return P_cur, {"stages": stages, "success": success, "history": {"positions": history_pos, "energies": history_E, "records": history_rec}}
        success = _grad_inf(sc, P_cur, cfg) <= gtol
        logger.info("Hybrid solver finished success=%s", success)
        return P_cur, {"stages": stages, "success": success, "history": {"positions": history_pos, "energies": history_E, "records": history_rec}}
    else:
        logger.info("Hybrid stage: Semi-Newton")
        current_stage = "semi"
        P1, info1 = solve_layout_semi_newton(scene, cfg, record=_rec_stage, energy_fn=energy_fn)
        _annotate(info1, "semi")
        stages.append(("semi", info1))
        _extend_history(info1, skip_first=False)
        if _grad_inf(scene, P1, cfg, energy_fn) <= gtol:
            logger.info("Hybrid solver finished after Semi-Newton stage")
            return P1, {"stages": stages, "success": True, "history": {"positions": history_pos, "energies": history_E, "records": history_rec}}
        sc = dict(scene)
        sc["labels_init"] = P1
        logger.info("Hybrid stage: L-BFGS")
        current_stage = "lbfgs"
        P2, info2 = solve_layout_lbfgs(sc, cfg, record=_rec_stage, energy_fn=energy_fn)
        _annotate(info2, "lbfgs")
        stages.append(("lbfgs", info2))
        _extend_history(info2, skip_first=True)
        success = _grad_inf(sc, P2, cfg, energy_fn) <= gtol
        logger.info("Hybrid solver finished success=%s", success)
        return P2, {"stages": stages, "success": success, "history": {"positions": history_pos, "energies": history_E, "records": history_rec}}
