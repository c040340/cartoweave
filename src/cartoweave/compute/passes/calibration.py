from __future__ import annotations

from typing import Any, Dict, Tuple
import inspect
import numpy as np

from cartoweave.utils.logging import logger
from cartoweave.compute.forces import enabled_terms, REGISTRY as FORCE_REGISTRY
from . import get_pass_cfg, REGISTRY as PASS_REGISTRY
from .base import ComputePass


# -----------------------------
# helpers
# -----------------------------

def _p_quantile(x: np.ndarray, clip_q: float, p_q: float) -> float:
    """Winsorised p_q-quantile of x."""
    if x.size == 0:
        return 0.0
    hi = float(np.quantile(x, clip_q))
    y = np.clip(x, 0.0, hi)
    return float(np.quantile(y, p_q))


def _force_entry(name: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Return (fn, defaults) from the forces registry.
    Registry entries may be stored as fn or (fn, defaults).
    """
    entry = FORCE_REGISTRY[name]
    if isinstance(entry, tuple):
        fn, defaults = entry
    else:
        fn, defaults = entry, {}
    return fn, dict(defaults)


def _term_cfg_from_public(cfg: Dict[str, Any], term: str) -> Dict[str, Any]:
    """
    Fetch per-term params from cfg.public.forces.<group>.<name> if present.
    Example term: "focus.attract" -> public.forces["focus"]["attract"]
    """
    try:
        grp, tname = term.split(".", 1)
    except ValueError:
        return {}
    return (
        ((cfg or {}).get("public", {}) or {})
        .get("forces", {})
        .get(grp, {})
        .get(tname, {}) or {}
    )


def _merged_params(cfg: Dict[str, Any], term: str) -> Dict[str, Any]:
    """Merge registry defaults with user overrides (public.forces.*).*"""
    _, defaults = _force_entry(term)
    user = _term_cfg_from_public(cfg, term)
    merged = dict(defaults)
    merged.update(user)
    return merged


def _scene_for_eval(scene: Dict[str, Any] | None, ctx: Dict[str, Any], P0: np.ndarray) -> Dict[str, Any]:
    """
    Build a scene dict suitable for force evaluation:
    - Ensure 'labels' exist and align with P0 rows
    - Thread through solver active ids if present
    """
    s: Dict[str, Any] = dict(scene or {})
    labels = s.get("labels")
    if not labels:
        labels = ctx.get("labels") or ctx.get("labels_curr") or ctx.get("labels_all") or []
        if labels:
            s["labels"] = labels

    # Thread active ids for consistency with solver code paths
    if "_active_ids_solver" not in s and "_active_ids_solver" in ctx:
        s["_active_ids_solver"] = ctx["_active_ids_solver"]

    # As a last resort, if still missing or length mismatched, fabricate lightweight labels
    if not s.get("labels") or len(s["labels"]) != int(P0.shape[0]):
        N = int(P0.shape[0])
        WH = s.get("WH") or ctx.get("WH")
        if isinstance(WH, np.ndarray) and WH.shape[0] == N:
            s["labels"] = [{"WH": tuple(map(float, WH[i]))} for i in range(N)]
        else:
            # Fallback: create empty dicts so readers relying only on length can proceed
            s["labels"] = [{} for _ in range(N)]
    return s


def _eval_force(name: str, scene: Dict[str, Any], P0: np.ndarray, cfg: Dict[str, Any], phase: str):
    """
    Call a force evaluate with a robust signature adapter:
      evaluate(scene, P, params, cfg[, phase=?])
    Only pass phase if the function supports it.
    """
    fn, _ = _force_entry(name)
    params = _merged_params(cfg, name)
    sig = inspect.signature(fn)
    if "phase" in sig.parameters:
        return fn(scene, P0, params, cfg, phase=phase)
    else:
        return fn(scene, P0, params, cfg)


# -----------------------------
# core calibration
# -----------------------------

def auto_calibrate_k(scene: Dict[str, Any] | None, P0: np.ndarray, cfg: Dict[str, Any], calib: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, float]:
    """Return scaling factors for term k-values based on observed force magnitudes."""
    if not calib.get("enable", False):
        return {}

    mags: Dict[str, np.ndarray] = {}

    scene_eval = _scene_for_eval(scene, ctx, P0)

    # Collect per-term magnitudes. If both stages are evaluated, the latter ("anchor")
    # intentionally overwrites the former to keep backward compatibility with code
    # that expects a single magnitude per term.
    for name in enabled_terms(cfg, phase="pre_anchor"):
        _, F, _ = _eval_force(name, scene_eval, P0, cfg, phase="pre_anchor")
        mags[name] = np.linalg.norm(F, axis=1).astype(float)

    for name in enabled_terms(cfg, phase="anchor"):
        _, F, _ = _eval_force(name, scene_eval, P0, cfg, phase="anchor")
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

        # Support both exact term match and group-level fallback (e.g., "focus")
        grp = term.split(".", 1)[0]
        goal_rel = float(target_rel.get(term, target_rel.get(grp, 1.0)))
        goal = goal_rel * base_obs
        ratio = goal / obs

        # Deadband to reduce churn
        if (1.0 - hyster) <= ratio <= (1.0 + hyster):
            continue

        s = max(clamp_min, min(clamp_max, ratio))
        scales[term] = s

    return scales


# -----------------------------
# Pass implementation
# -----------------------------

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
        scales = auto_calibrate_k(scene, P, cfg, conf, ctx)
        if not scales:
            return

        alpha = float(conf.get("ema_alpha", self.ema_alpha))

        # We update k_* values in cfg.public.forces.<group>.<name> to match the rest of the project.
        forces_cfg = cfg.setdefault("public", {}).setdefault("forces", {})

        for term, scale in scales.items():
            try:
                grp, tname = term.split(".", 1)
            except ValueError:
                continue

            term_cfg = (forces_cfg.get(grp) or {}).get(tname)
            if not isinstance(term_cfg, dict):
                continue

            k_keys = [k for k in term_cfg.keys() if k.startswith("k_")]
            if not k_keys:
                continue

            for kk in k_keys:
                k_old = term_cfg.get(kk)
                if k_old is None:
                    continue

                prev = self.prev_k.get(f"{term}:{kk}", float(k_old))
                k_new = float(k_old) * float(scale)
                if alpha < 1.0:
                    k_new = (1.0 - alpha) * prev + alpha * k_new

                term_cfg[kk] = float(k_new)

                # Emit an event so the recorder can attach it to frame.meta.events
                if hasattr(self, "pm") and self.pm is not None:
                    self.pm.emit_event({
                        "pass": "calibration",
                        "info": "k_update",
                        "term": term,
                        "param": kk,
                        "k_old": float(k_old),
                        "k_new": float(k_new),
                        "scale": float(scale),
                        "ema_alpha": float(alpha),
                        "global_iter": getattr(self.pm, "eval_index", 0),
                    })

                self.prev_k[f"{term}:{kk}"] = float(k_new)
            logger.info(f"[calibration] term={term} k={k_new:.3g}")


# Register into the PASSES registry (not forces)
PASS_REGISTRY["calibration"] = (CalibrationPass, {"ema_alpha": 1.0})

__all__ = ["CalibrationPass", "auto_calibrate_k"]