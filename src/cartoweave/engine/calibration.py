"""Calibration utilities for force terms.

The solver exposes many force terms whose shape parameters (``p``, ``alpha``,
``beta`` …) remain mostly static during an action.  Their linear scale ``k``
controls the relative strength and can be adjusted between actions without
changing the optimisation problem's dimensionality.  This module implements a
light‑weight calibration stage that observes current force magnitudes and
suggests new ``k`` values before the solver iterates.

Two concepts are distinguished:

``apply_shape_profile``
    Apply a pre-defined profile of shape parameters.  This happens rarely and
    never touches ``k``.
``auto_calibrate_k``
    Observe all active force terms at the initial configuration ``P0`` and
    rescale their ``k`` values to meet target ratios relative to a base term.
    The observation is performed once per action.

Additionally helpers for crowding estimation, calibration triggering and a
simple exponential moving average for temporally smoothing ``k`` values are
provided.
"""
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np

from .forces import REGISTRY, enabled_terms
from .logs import log

# ---------------------------------------------------------------------------
# Shape profiles -------------------------------------------------------------

_SHAPE_PROFILES: Dict[str, Dict[str, float]] = {
    "sparse": {},
    "default": {},
    "dense": {},
}


def _shape_profile_dict(name: str, sigma_scale: float) -> Dict[str, float]:
    """Return a dictionary of shape parameters for ``name``.

    This simplified implementation only ships placeholder profiles used in the
    tests.  Real deployments would populate the profile dictionaries with
    parameters such as softplus ``beta`` values or smooth ``epsilon`` terms.
    ``sigma_scale`` can be used to uniformly scale any ``*.sigma.*`` entries.
    """

    base = _SHAPE_PROFILES.get(name, _SHAPE_PROFILES.get("default", {}))
    out: Dict[str, float] = {}
    for k, v in base.items():
        if "sigma" in k:
            out[k] = float(v) * float(sigma_scale)
        else:
            out[k] = float(v)
    return out


def apply_shape_profile(cfg: Dict[str, Any], logger=None) -> Dict[str, float]:
    """Apply a shape profile to ``cfg`` in-place.

    The profile is applied at most once unless ``calib.shape.scope`` is set to
    ``"always"``.  Only keys already present in ``cfg`` are updated which keeps
    the operation backwards compatible.
    """

    if not cfg.get("calib.shape.enable", True):
        return {}
    scope = cfg.get("calib.shape.scope", "once").lower()
    if scope == "once" and cfg.get("_shape_applied"):
        return {}

    name = (cfg.get("calib.shape.name", "default") or "default").lower()
    scale = float(cfg.get("calib.shape.sigma_scale", 1.0))
    patch = _shape_profile_dict(name, scale)
    for k, v in patch.items():
        if k in cfg:
            cfg[k] = v
    cfg["_shape_applied"] = True
    log(logger, f"[shape] profile={name} sigma={scale} applied={len(patch)}")
    return patch


# ---------------------------------------------------------------------------
# Crowding & triggering ------------------------------------------------------


def crowding_score(scene: Dict[str, Any], P0: np.ndarray, cfg: Dict[str, Any]) -> float:
    """Return a lightweight crowding score in ``[0, 1]``.

    The heuristic combines three proxies: occupied area ratio, inverse nearest
    neighbour distance and a bounding-box overlap ratio.
    """

    P0 = np.asarray(P0, float)
    WH = np.asarray(scene.get("WH", np.zeros_like(P0)), float)
    W, H = scene.get("frame_size", (1.0, 1.0))

    area = float((WH[:, 0] * WH[:, 1]).sum())
    dens = np.clip(area / (float(W) * float(H) + 1e-12), 0.0, 1.0)

    if P0.shape[0] > 1:
        d = P0[:, None, :] - P0[None, :, :]
        d = np.hypot(d[..., 0], d[..., 1])
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        p25 = float(np.percentile(nn, 25)) if nn.size else 0.0
        invnn = 1.0 / (p25 + 1e-6)
        invnn = float(np.clip(invnn / (max(W, H) ** -1 + 1e-12), 0.0, 1.0))
    else:
        invnn = 0.0

    overlap = 0.0
    if P0.shape[0] > 1 and WH.shape[0] >= P0.shape[0]:
        cnt = 0
        hit = 0
        for i in range(P0.shape[0]):
            xi, yi = P0[i]
            wi, hi = WH[i]
            for j in range(i + 1, P0.shape[0]):
                xj, yj = P0[j]
                wj, hj = WH[j]
                if abs(xi - xj) < 0.5 * (wi + wj) and abs(yi - yj) < 0.5 * (hi + hj):
                    hit += 1
                cnt += 1
        overlap = hit / cnt if cnt else 0.0

    w1, w2, w3 = 0.4, 0.4, 0.2
    C = w1 * dens + w2 * invnn + w3 * overlap
    return float(np.clip(C, 0.0, 1.0))


def should_recalibrate_k(
    C: float,
    C_prev: float | None,
    act_metrics: Dict[str, Any],
    step: Dict[str, Any],
    cfg: Dict[str, Any],
) -> bool:
    """Decide whether ``auto_calibrate_k`` should run for this action."""

    if not cfg.get("calib.trigger.enable", True):
        return True
    tau_on = cfg.get("calib.trigger.tau_on", 0.60)
    tau_off = cfg.get("calib.trigger.tau_off", 0.45)
    delta = cfg.get("calib.trigger.delta", 0.15)

    big_change = bool(step.get("struct_change"))
    if C_prev is not None:
        big_change = big_change or abs(C - C_prev) > delta

    if C_prev is None:
        return big_change or C >= tau_on
    if C >= tau_on and big_change:
        return True
    if C <= tau_off:
        return False
    return False


# ---------------------------------------------------------------------------
# k calibration --------------------------------------------------------------

TERM2K: Dict[str, List[str]] = {
    "ll.rect": ["ll.k.repulse"],
    "ll.disk": ["ll.k.repulse"],
    "boundary.wall": ["boundary.k.wall"],
    "anchor.spring": ["anchor.k.spring"],
    "pl.rect": ["pl.k.repulse"],
    "ln.rect": ["ln.k.repulse"],
    "area.embed": ["area.k.embed", "area.k.tan"],
    "area.cross": ["area.k.cross"],
    "area.softout": ["area.k.softout"],
    "focus.attract": ["focus.k.attract"],
}

TARGET_REL: Dict[str, Dict[str, float]] = {
    "sparse": {
        "area.cross": 6,
        "ll.rect": 3,
        "ll.disk": 3,
        "boundary.wall": 3,
        "area.embed": 2.5,
        "anchor.spring": 2.2,
        "ln.rect": 1.8,
        "pl.rect": 1.6,
        "area.softout": 1.1,
        "focus.attract": 1,
    },
    "default": {
        "area.cross": 8,
        "ll.rect": 5,
        "ll.disk": 5,
        "boundary.wall": 4,
        "area.embed": 3.5,
        "anchor.spring": 3.0,
        "ln.rect": 2.5,
        "pl.rect": 2.0,
        "area.softout": 1.2,
        "focus.attract": 1,
    },
    "dense": {
        "area.cross": 10,
        "ll.rect": 7,
        "ll.disk": 7,
        "boundary.wall": 5,
        "area.embed": 4.2,
        "anchor.spring": 3.5,
        "ln.rect": 3.0,
        "pl.rect": 2.5,
        "area.softout": 1.5,
        "focus.attract": 1,
    },
}


def auto_calibrate_k(
    scene: Dict[str, Any],
    P0: np.ndarray,
    cfg: Dict[str, Any],
    logger=None,
) -> Dict[str, float]:
    """Return calibrated ``k`` values based on force observations at ``P0``."""

    if not cfg.get("calib.k.enable", False):
        return {}

    P0 = np.asarray(P0, float)
    mags: Dict[str, np.ndarray] = {}
    for name in enabled_terms(cfg, phase="pre_anchor"):
        _, F, _ = REGISTRY[name](scene, P0, cfg, phase="pre_anchor")
        mags[name] = np.linalg.norm(F, axis=1).astype(float)
    for name in enabled_terms(cfg, phase="anchor"):
        _, F, _ = REGISTRY[name](scene, P0, cfg, phase="anchor")
        mags[name] = np.linalg.norm(F, axis=1).astype(float)

    clip_q = cfg.get("calib.k.clip_q", 0.995)
    p_q = cfg.get("calib.k.p_q", 0.95)
    base = cfg.get("calib.k.base_term", "focus.attract")

    def p_quantile(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        hi = np.quantile(x, clip_q)
        return float(np.quantile(np.clip(x, 0, hi), p_q))

    base_obs = p_quantile(mags.get(base, np.zeros(0)))
    if base_obs <= 1e-12:
        base_obs = p_quantile(mags.get("boundary.wall", np.zeros(0)))
    if base_obs <= 1e-12 and mags:
        base_obs = float(np.median([p_quantile(v) for v in mags.values() if v.size]))

    profile = (cfg.get("calib.k.profile", "default") or "default").lower()
    target_rel = TARGET_REL.get(profile, TARGET_REL["default"])

    min_act = cfg.get("calib.k.min_act", 0.05)
    hysteresis = cfg.get("calib.k.hysteresis", 0.25)
    cmin = cfg.get("calib.k.clamp_min", 0.33)
    cmax = cfg.get("calib.k.clamp_max", 3.0)
    act_thresh = 0.05 * base_obs

    k_hat: Dict[str, float] = {}
    for term, v in mags.items():
        if term == base or v.size == 0:
            continue
        act_ratio = float((v > act_thresh).sum() / v.size)
        if act_ratio < min_act:
            continue
        obs = p_quantile(v)
        if obs <= 1e-12:
            continue
        goal = float(
            target_rel.get(term, target_rel.get(term.split(".")[0], 1.0))
        ) * base_obs
        ratio = goal / obs
        if (1.0 - hysteresis) <= ratio <= (1.0 + hysteresis):
            continue
        s = float(max(cmin, min(cmax, ratio)))
        for kname in TERM2K.get(term, []):
            if kname in cfg:
                k_hat[kname] = cfg[kname] * s
                log(logger, f"[autocal] {term:<14} {kname} x{s:4.2f} obs={obs:.3g} goal={goal:.3g}")

    return k_hat


# ---------------------------------------------------------------------------
# EMA update ----------------------------------------------------------------


def ema_update_k(
    prev_k: Dict[str, float] | None,
    new_k: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    """Blend ``new_k`` into ``prev_k`` using exponential moving average."""

    if not new_k:
        return prev_k or {}
    out = dict(prev_k or {})
    for kname, v_new in new_k.items():
        v_old = out.get(kname, v_new)
        out[kname] = (1.0 - alpha) * v_old + alpha * v_new
    return out


__all__ = [
    "apply_shape_profile",
    "auto_calibrate_k",
    "crowding_score",
    "should_recalibrate_k",
    "ema_update_k",
    "TERM2K",
    "TARGET_REL",
]
