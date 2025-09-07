"""Matplotlib backend selection utilities."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)


def setup_matplotlib_backend(prefer: str = "TkAgg", fallback: str = "Agg") -> str:
    """Try to set a Matplotlib backend.

    The function first honours ``MATPLOTLIB_BACKEND`` when set.  If that
    fails, it attempts to use ``prefer`` and finally ``fallback``.  The backend
    string actually in use is returned.
    """

    import matplotlib

    chosen: str | None = None
    env_backend = os.environ.get("MATPLOTLIB_BACKEND", "").strip()
    if env_backend:
        try:
            matplotlib.use(env_backend)
            chosen = env_backend
        except Exception:  # pragma: no cover - log only
            logger.warning(
                "Failed to use MATPLOTLIB_BACKEND=%s; will try %s→%s",
                env_backend,
                prefer,
                fallback,
            )

    if chosen is None:
        try:
            matplotlib.use(prefer)
            chosen = prefer
        except Exception:  # pragma: no cover - rare
            try:
                matplotlib.use(fallback)
                chosen = fallback
            except Exception:  # pragma: no cover - extremely rare
                chosen = matplotlib.get_backend()
                logger.warning(
                    "Could not set backend to %s/%s. Using default: %s",
                    prefer,
                    fallback,
                    chosen,
                )

    logger.info("Matplotlib backend = %s", chosen)
    return chosen


def use_compatible_backend() -> str:
    """Backward compatible alias for :func:`setup_matplotlib_backend`."""

    return setup_matplotlib_backend()


__all__ = ["setup_matplotlib_backend", "use_compatible_backend"]

