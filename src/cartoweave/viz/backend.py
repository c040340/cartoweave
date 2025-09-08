from __future__ import annotations

import os
import sys
import importlib
from typing import Optional


def _has_display() -> bool:
    """Best-effort check for a GUI display."""
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return False
    return True


def _tk_available() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except Exception:
        return False


def detect_backend(prefer: str = "TkAgg") -> str:
    """Decide which matplotlib backend to use."""
    env = os.environ.get("MPLBACKEND")
    if env:
        return env
    if prefer.lower() in ("tkagg", "tk"):
        if _tk_available() and _has_display():
            return "TkAgg"
    return "Agg"


def setup_matplotlib_backend(
    prefer: str = "TkAgg", fallback: str = "Agg", force: Optional[str] = None
) -> str:
    """Set matplotlib backend BEFORE importing pyplot.

    Safe to call multiple times; if pyplot already imported, do not change backend.
    Returns the backend string in use.
    """
    if "matplotlib.pyplot" in sys.modules:
        import matplotlib
        return matplotlib.get_backend()

    backend = force or detect_backend(prefer=prefer)
    if force is None and backend != prefer:
        backend = fallback

    if force:
        os.environ["MPLBACKEND"] = force
    elif "MPLBACKEND" not in os.environ:
        os.environ["MPLBACKEND"] = backend

    import matplotlib
    try:
        matplotlib.use(os.environ.get("MPLBACKEND", backend), force=True)
    except Exception:
        matplotlib.use("Agg", force=True)

    importlib.import_module("matplotlib.pyplot")
    return matplotlib.get_backend()


__all__ = ["detect_backend", "setup_matplotlib_backend"]

