# -*- coding: utf-8 -*-
"""Shared dataclasses and helpers for the compute module."""

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from cartoweave.contracts.viewpack import Frame, ViewPack

Array2 = np.ndarray  # shape == (L, 2)

# Energy callback signature: ``(P, scene, mask, cfg) -> (E, G, comps, meta)``
EnergyFn = Callable[
    [Array2, object, np.ndarray, dict],
    Tuple[float, Array2, Dict[str, Array2], dict],
]


def _grad_metrics(G: Optional[Array2]) -> Dict[str, float]:
    """Return infinity and L2 norms of ``G`` if provided."""

    import numpy as np

    if G is None:
        return {}
    return {
        "g_inf": float(np.linalg.norm(G, ord=np.inf)),
        "g_norm": float(np.linalg.norm(G)),
    }


__all__ = ["Frame", "ViewPack", "Array2", "EnergyFn", "_grad_metrics"]
