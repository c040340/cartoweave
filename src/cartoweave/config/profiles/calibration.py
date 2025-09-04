from __future__ import annotations
from typing import Dict, Any

# Term-group keys are strings; engine will interpret them. Keep them human-friendly.
# Do not include k-key paths here; these are strategy-level knobs.
CALIB_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "target_rel": {
            "area.cross": 8.0,
            "ll.rect": 5.0,
            "ll.disk": 5.0,
            "boundary.wall": 4.0,
            "area.embed": 3.5,
            "anchor.spring": 3.0,
            "ln.rect": 2.5,
            "pl.rect": 2.0,
            "area.softout": 1.2,
            "focus.attract": 1.0,
        },
        "trigger":   {"tau_on": 0.60, "tau_off": 0.45, "delta": 0.15},
        "quantiles": {"clip_q": 0.995, "p_q": 0.95},
        "limits":    {"clamp_min": 0.33, "clamp_max": 3.0, "min_act": 0.05},
        "ema_alpha": 0.30,
    },
    "dense": {
        "target_rel": {
            "area.cross": 10.0, "ll.rect": 7.0, "ll.disk": 7.0, "boundary.wall": 5.0,
            "area.embed": 4.2,  "anchor.spring": 3.5, "ln.rect": 3.0,
            "pl.rect": 2.5,     "area.softout": 1.5,  "focus.attract": 1.0,
        },
        "trigger":   {"tau_on": 0.65, "tau_off": 0.50, "delta": 0.15},
        "quantiles": {"clip_q": 0.995, "p_q": 0.95},
        "limits":    {"clamp_min": 0.33, "clamp_max": 3.0, "min_act": 0.05},
        "ema_alpha": 0.35,
    },
    "sparse": {
        "target_rel": {
            "area.cross": 6.0, "ll.rect": 3.0, "ll.disk": 3.0, "boundary.wall": 3.0,
            "area.embed": 2.5, "anchor.spring": 2.2, "ln.rect": 1.8,
            "pl.rect": 1.6,    "area.softout": 1.1,  "focus.attract": 1.0,
        },
        "trigger":   {"tau_on": 0.55, "tau_off": 0.40, "delta": 0.15},
        "quantiles": {"clip_q": 0.995, "p_q": 0.95},
        "limits":    {"clamp_min": 0.33, "clamp_max": 3.0, "min_act": 0.05},
        "ema_alpha": 0.25,
    },
}
