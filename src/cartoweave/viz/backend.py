"""Matplotlib backend selection helpers."""

from __future__ import annotations

import os
import sys
import matplotlib


def use_compatible_backend() -> None:
    """Select a usable Matplotlib backend for the current environment.

    Priority:
    1. Environment variable ``MPLBACKEND`` if set.
    2. On macOS, try interactive backends (``TkAgg``, ``MacOSX``).
    3. Fallback to the non-interactive ``Agg`` backend.
    """
    env_backend = os.environ.get("MPLBACKEND")
    if env_backend:
        try:
            matplotlib.use(env_backend, force=True)
            return
        except Exception:
            pass

    if sys.platform.startswith("darwin"):
        for cand in ("TkAgg", "MacOSX"):
            try:
                matplotlib.use(cand, force=True)
                return
            except Exception:
                continue

    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        # If even Agg fails, let Matplotlib fall back to its default.
        pass
