from __future__ import annotations
from typing import Dict, Any

def base_defaults() -> Dict[str, Any]:
    """
    Minimal defaults that are safe across environments.
    Keep this very small; do not add feature toggles here.
    """
    calib_defaults = {
        # ---- shape profile gating ----
        "calib.shape.enable": False,
        "calib.shape.name": "default",
        "calib.shape.sigma_scale": 1.0,

        # ---- k-only calibration gating ----
        "calib.k.enable": False,              # KEEP OFF by default
        "calib.k.profile": "default",
        "calib.k.base_term": "focus.attract",

        # quantiles & activity thresholds
        "calib.k.clip_q": 0.995,
        "calib.k.p_q": 0.95,
        "calib.k.min_act": 0.05,

        # hysteresis & scaling clamps
        "calib.k.hysteresis": 0.25,
        "calib.k.clamp_min": 0.33,
        "calib.k.clamp_max": 3.0,

        # temporal smoothing for per-action k
        "calib.k.ema_alpha": 0.3,

        # ---- triggering (crowding/hysteresis) ----
        "calib.trigger.enable": True,
        "calib.trigger.tau_on": 0.60,
        "calib.trigger.tau_off": 0.45,
        "calib.trigger.delta": 0.15,
    }

    d = {
        # Logging defaults (exists in SPEC and safe at runtime)
        "logging.level": "INFO",

        # Example numeric knobs commonly needed by callers;
        # these keys may not be in SPEC yet (that's OK).
        "engine.max_iter_hint": 200,          # hint, not enforced
        "engine.bounds.padding_px": 4.0,      # generic padding
    }
    d.update(calib_defaults)
    return d
