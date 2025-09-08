from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cartoweave.utils.logging import logger
from cartoweave.compute.forces import REGISTRY, enabled_terms

from .base import ComputePass
from . import get_pass_cfg


def _p_quantile(x: np.ndarray, clip_q: float, p_q: float) -> float:
    """Winsorised ``p_q``-quantile of ``x``."""

    if x.size == 0:
        return 0.0
    hi = float(np.quantile(x, clip_q))
    y = np.clip(x, 0.0, hi)
    return float(np.quantile(y, p_q))


def auto_calibrate_k(scene: Dict[str, Any], P0: np.ndarray, cfg: Dict[str, Any], calib: Dict[str, Any]) -> Dict[str, float]:
    """Return scaling factors for term ``k`` values based on observed forces."""

    if not calib.get("enable", False):
        return {}

    mags: Dict[str, np.ndarray] = {}
    for name in enabled_terms(cfg, phase="pre_anchor"):
        _, F, _ = REGISTRY[name](scene, P0, cfg, phase="pre_anchor")
        mags[name] = np.linalg.norm(F, axis=1).astype(float)
    for name in enabled_terms(cfg, phase="anchor"):
        _, F, _ = REGISTRY[name](scene, P0, cfg, phase="anchor")
        mags[name] = np.linalg.norm(F, axis=1).astype(float)

    clip_q = float(calib.get("clip_q", 0.995))
    p_q = float(calib.get("p_q", 0.95))
    clamp_min = float(calib.get("clamp_min", 0.33))
    clamp_max = float(calib.get("clamp_max", 3.0))
    min_act = float(calib.get("min_act", 0.05))
    hyster = float(calib.get("hysteresis", 0.25))
    base_term = calib.get("base_term", "focus.attract")
    target_rel = calib.get("target_rel", {}) or {}

    def p_quant(term: str) -> float:
        return _p_quantile(mags.get(term, np.zeros(0)), clip_q, p_q)

    base_obs = p_quant(base_term)
    if base_obs <= 1e-12 and mags:
        vals = [p_quant(t) for t in mags.keys()]
        nz = [v for v in vals if v > 1e-12]
        base_obs = float(np.median(nz)) if nz else 1.0
    act_thresh = 0.05 * base_obs

    scales: Dict[str, float] = {}
    for term, vec in mags.items():
        if vec.size == 0:
            continue
        act_ratio = float((vec > act_thresh).sum() / vec.size)
        if act_ratio < min_act:
            continue
        obs = _p_quantile(vec, clip_q, p_q)
        if obs <= 1e-12:
            continue
        goal_rel = float(target_rel.get(term, target_rel.get(term.split(".")[0], 1.0)))
        goal = goal_rel * base_obs
        ratio = goal / obs
        if (1.0 - hyster) <= ratio <= (1.0 + hyster):
            continue
        s = max(clamp_min, min(clamp_max, ratio))
        scales[term] = s
    return scales


class CalibrationPass(ComputePass):
    """Adaptive force balancing executed before each action."""

    name = "calibration"

    def __init__(self, ema_alpha: float = 1.0):
        self.ema_alpha = float(ema_alpha)
        self.prev_k: Dict[str, float] = {}

    def run(self, ctx: Dict[str, Any]) -> None:  # pragma: no cover - thin glue
        cfg = getattr(getattr(self, "pm", None), "cfg", {}) or {}
        conf = get_pass_cfg(
            cfg,
            "calibration",
            {
                "enable": False,
                "clip_q": 0.995,
                "p_q": 0.95,
                "clamp_min": 0.33,
                "clamp_max": 3.0,
                "min_act": 0.05,
                "hysteresis": 0.25,
                "ema_alpha": self.ema_alpha,
                "base_term": "focus.attract",
                "target_rel": {},
            },
        )
        if not conf.get("enable", False):
            return

        scene = ctx.get("scene") or {}
        P = np.asarray(ctx.get("P"), float)
        scales = auto_calibrate_k(scene, P, cfg, conf)
        if not scales:
            return

        alpha = float(conf.get("ema_alpha", self.ema_alpha))
        forces_cfg = (cfg.setdefault("public", {}).setdefault("forces", {}))
        for term, scale in scales.items():
            grp, tname = term.split(".")
            term_cfg = ((forces_cfg.get(grp) or {}).get(tname))
            if not isinstance(term_cfg, dict):
                continue
            k_old = term_cfg.get("k")
            if k_old is None:
                continue
            prev = self.prev_k.get(term, float(k_old))
            k_new = float(k_old) * float(scale)
            if alpha < 1.0:
                k_new = (1.0 - alpha) * prev + alpha * k_new
            term_cfg["k"] = float(k_new)
            self.prev_k[term] = float(k_new)
            logger.info(f"[calibration] term={term} k={k_new:.3g}")


REGISTRY["calibration"] = (CalibrationPass, {"ema_alpha": 1.0})

__all__ = ["CalibrationPass", "auto_calibrate_k"]
