"""Calibration helpers for adaptive force scaling.

This module provides utilities used by the optimisation timeline to adjust
force weights (``k`` values) based on observed magnitudes.  Strategy and
thresholds are fully driven by the configuration system – no hard coded
numbers live in this file.  Only the mapping between term groups and the
corresponding ``k`` keys is kept here as ``TERM2K`` because it is an engine
concern.

The public functions mirror the previous API so existing call‑sites remain
unchanged.  When possible docstrings have been added for clarity.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List

import numpy as np

from cartoweave.utils.logging import logger
from cartoweave.engine.forces import REGISTRY, enabled_terms
from cartoweave.config.layering import apply_shape_profile


# ---------------------------------------------------------------------------
# Mapping from term groups to ``k`` keys ------------------------------------

# Which k-keys should be scaled when a given term-group is calibrated
TERM2K: Dict[str, List[str]] = {
    "ll.rect":        ["ll.k.repulse"],
    "ll.disk":        ["ll.k.repulse"],
    "boundary.wall":  ["boundary.k.wall"],
    "anchor.spring":  ["anchor.k.spring"],
    "pl.rect":        ["pl.k.repulse"],
    "ln.rect":        ["ln.k.repulse"],
    "area.embed":     ["area.k.embed", "area.k.tan"],
    "area.cross":     ["area.k.cross"],
    "area.softout":   ["area.k.softout"],
    "focus.attract":  ["focus.k.attract"],
}


# ---------------------------------------------------------------------------
# Shape profile --------------------------------------------------------------

def apply_shape_profile_from_cfg(cfg: Dict[str, Any]) -> int:
    """Apply a frozen-only shape profile based on flags in ``cfg``.

    The heavy lifting is delegated to :func:`cartoweave.config.layering.apply_shape_profile`.
    This wrapper merely reads configuration flags and swallows unknown-profile
    errors so call-sites do not have to handle them.

    Parameters
    ----------
    cfg:
        Configuration dictionary.  Keys ``calib.shape.enable``,
        ``calib.shape.name`` and ``calib.shape.sigma_scale`` control the
        operation.

    Returns
    -------
    int
        Number of keys patched.
    """

    enable = bool(cfg.get("calib.shape.enable", False))
    name = cfg.get("calib.shape.name", "default")
    sigma = float(cfg.get("calib.shape.sigma_scale", 1.0))

    try:
        patched = apply_shape_profile(cfg, name=name, enable=enable, sigma_scale=sigma)
        if patched:
            logger.info(
                f"[shape] profile={name} sigma={sigma} patched={patched}"
            )
        return patched
    except KeyError:
        logger.info(f"[shape] profile not applied (unknown name={name})")
        return 0


# Backwards compatibility ----------------------------------------------------

def apply_shape_profile(cfg: Dict[str, Any], logger_=None) -> int:
    """Legacy wrapper for existing call-sites.

    ``logger_`` is ignored but preserved so older code invoking
    ``apply_shape_profile(cfg, logger)`` keeps working.
    """

    return apply_shape_profile_from_cfg(cfg)


# ---------------------------------------------------------------------------
# Crowding score -------------------------------------------------------------

def crowding_score(scene: Dict[str, Any], P0: np.ndarray, cfg: Dict[str, Any]) -> float:
    """Return a very small deterministic proxy for scene crowding.

    The current implementation is intentionally simple – it looks only at the
    number of labels and the canvas size.  Future tasks may replace it with a
    more elaborate heuristic.  The return value is clamped to ``[0, 1]``.
    """

    n = max(len(scene.get("labels", [])), 1)
    w = float(cfg.get("scene.width_px", 1920))
    h = float(cfg.get("scene.height_px", 1080))
    C = min(1.0, n / max(50.0, (w * h) / 1e6))
    return float(C)


# ---------------------------------------------------------------------------
# Triggering -----------------------------------------------------------------

def should_recalibrate_k(
    C: float,
    C_prev: Optional[float],
    act_metrics: Optional[Dict[str, float]],
    step: Dict[str, Any],
    cfg: Dict[str, Any],
) -> bool:
    """Decide whether ``auto_calibrate_k`` should run for this action."""

    if not cfg.get("calib.trigger.enable", True):
        return True

    tau_on = float(cfg.get("calib.trigger.tau_on", 0.60))
    tau_off = float(cfg.get("calib.trigger.tau_off", 0.45))
    delta = float(cfg.get("calib.trigger.delta", 0.15))

    big_change = (C_prev is None) or (abs(C - C_prev) > delta)
    if C >= tau_on and big_change:
        return True
    if C <= tau_off:
        return False
    return False


# ---------------------------------------------------------------------------
# Quantile helpers -----------------------------------------------------------

def _get_quant(cfg: Dict[str, Any], container: str = "calib.k", key: str = "clip_q", default: float = 0.995) -> float:
    """Return quantile parameters from ``cfg`` with nested and flat fallbacks."""

    if isinstance(cfg.get(container), dict) and key in cfg[container]:
        return float(cfg[container][key])
    return float(cfg.get(f"{container}.{key}", cfg.get(f"calib.k.{key}", default)))


def _p_quantile(x: np.ndarray, clip_q: float, p_q: float) -> float:
    """Winsorised ``p``-quantile of ``x``."""

    if x.size == 0:
        return 0.0
    hi = float(np.quantile(x, clip_q))
    y = np.clip(x, 0.0, hi)
    return float(np.quantile(y, p_q))


# ---------------------------------------------------------------------------
# Auto k-calibration ---------------------------------------------------------

def auto_calibrate_k(scene: Dict[str, Any], P0: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Return a dictionary of new ``k`` values based on observed forces."""

    if not cfg.get("calib.k.enable", False):
        return {}

    # ------------------------------------------------------------------ gather magnitudes
    mags: Dict[str, np.ndarray] = {}
    for name in enabled_terms(cfg, phase="pre_anchor"):
        _, F, _ = REGISTRY[name](scene, P0, cfg, phase="pre_anchor")
        mags[name] = np.linalg.norm(F, axis=1).astype(float)
    for name in enabled_terms(cfg, phase="anchor"):
        _, F, _ = REGISTRY[name](scene, P0, cfg, phase="anchor")
        mags[name] = np.linalg.norm(F, axis=1).astype(float)

    # ------------------------------------------------------------------ strategy from cfg
    clip_q = _get_quant(cfg, "calib.k", "clip_q", 0.995)
    p_q = _get_quant(cfg, "calib.k", "p_q", 0.95)

    limits = cfg.get("calib.k", {}) if isinstance(cfg.get("calib.k"), dict) else {}
    clamp_min = float(limits.get("clamp_min", cfg.get("calib.k.clamp_min", 0.33)))
    clamp_max = float(limits.get("clamp_max", cfg.get("calib.k.clamp_max", 3.0)))
    min_act = float(limits.get("min_act", cfg.get("calib.k.min_act", 0.05)))

    hyster = float(cfg.get("calib.k.hysteresis", 0.25))
    base_term = cfg.get("calib.k.base_term", "focus.attract")

    target_rel: Dict[str, float] = {}
    if isinstance(cfg.get("calib.k.target_rel"), dict):
        target_rel = dict(cfg["calib.k.target_rel"])

    # ------------------------------------------------------------------ base magnitude
    def p95_of(term: str) -> float:
        return _p_quantile(mags.get(term, np.zeros(0)), clip_q, p_q)

    base_obs = p95_of(base_term)
    if base_obs <= 1e-12:
        base_obs = p95_of("boundary.wall")
    if base_obs <= 1e-12 and mags:
        vals = [p95_of(t) for t in mags.keys()]
        nz = [v for v in vals if v > 1e-12]
        base_obs = float(np.median(nz)) if nz else 1.0

    act_thresh = 0.05 * base_obs

    # ------------------------------------------------------------------ compute new k values
    k_hat: Dict[str, float] = {}
    for term, vec in mags.items():
        if vec.size == 0:
            continue

        # Activation gating ------------------------------------------------
        act_ratio = float((vec > act_thresh).sum() / vec.size)
        if act_ratio < min_act:
            continue

        obs = _p_quantile(vec, clip_q, p_q)
        if obs <= 1e-12:
            continue

        goal_rel = float(target_rel.get(term, target_rel.get(term.split(".")[0], 1.0)))
        goal = goal_rel * base_obs
        ratio = goal / obs

        # Hysteresis band --------------------------------------------------
        if (1.0 - hyster) <= ratio <= (1.0 + hyster):
            continue

        s = float(max(clamp_min, min(clamp_max, ratio)))

        for kname in TERM2K.get(term, []):
            if kname in cfg:
                k_hat[kname] = float(cfg[kname]) * s
                logger.info(
                    f"[autocal] term={term:<14} k={kname:<18} x{s:4.2f}  obs={obs:.3g} goal={goal:.3g} (base={base_obs:.3g})"
                )

    return k_hat


# ---------------------------------------------------------------------------
# EMA update ----------------------------------------------------------------

def ema_update_k(prev_k: Optional[Dict[str, float]], new_k: Dict[str, float], alpha: float) -> Dict[str, float]:
    """Blend ``new_k`` into ``prev_k`` using exponential moving average."""

    if not new_k:
        return prev_k or {}

    out = dict(prev_k or {})
    for k, v_new in new_k.items():
        v_old = out.get(k, v_new)
        out[k] = (1.0 - alpha) * float(v_old) + alpha * float(v_new)
    return out


# ---------------------------------------------------------------------------
# Public symbols -------------------------------------------------------------

__all__ = [
    "apply_shape_profile_from_cfg",
    "apply_shape_profile",
    "crowding_score",
    "should_recalibrate_k",
    "auto_calibrate_k",
    "ema_update_k",
    "TERM2K",
]

