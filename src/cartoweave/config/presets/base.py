from __future__ import annotations
from typing import Dict, Any

def base_defaults() -> Dict[str, Any]:
    """
    Minimal defaults that are safe across environments.
    Keep this very small; do not add feature toggles here.
    """
    return {
        # Logging defaults (exists in SPEC and safe at runtime)
        "logging.level": "INFO",

        # Example numeric knobs commonly needed by callers;
        # these keys may not be in SPEC yet (that's OK).
        "engine.max_iter_hint": 200,          # hint, not enforced
        "engine.bounds.padding_px": 4.0,      # generic padding
    }
