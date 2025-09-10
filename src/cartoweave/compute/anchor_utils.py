from __future__ import annotations

from typing import Any

import numpy as np


def label_anchor_xy(label: Any) -> np.ndarray:
    """Return precomputed anchor coordinate for ``label``.

    The coordinate is expected to be supplied during data generation and stored
    either in ``label.meta['anchor_xy']`` or ``label.anchor.xy``.  When neither
    is present the function returns ``(nan, nan)``.
    """

    if isinstance(label, dict):
        meta = label.get("meta") or {}
        xy = meta.get("anchor_xy")
        if xy is not None:
            return np.asarray(xy, float)
        a = label.get("anchor") or {}
        xy = a.get("xy")
        if xy is not None:
            return np.asarray(xy, float)
    else:
        meta = getattr(label, "meta", {}) or {}
        xy = meta.get("anchor_xy")
        if xy is not None:
            return np.asarray(xy, float)
        a = getattr(label, "anchor", None)
        if a is not None and getattr(a, "xy", None) is not None:
            x, y = a.xy
            return np.asarray([x, y], float)
    return np.array([np.nan, np.nan], float)


__all__ = ["label_anchor_xy"]
