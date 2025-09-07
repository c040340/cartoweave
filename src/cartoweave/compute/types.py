# -*- coding: utf-8 -*-
"""Shared dataclasses and helpers for the compute module."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Re-export the strict ViewPack contract for downstream modules.
from cartoweave.contracts.viewpack_v1 import (
    ViewPack,
    VPFrame,
    VPPass,
    VPSources,
)

Array2 = np.ndarray  # shape == (L, 2)


# ---------------------------------------------------------------------------
# Typed containers exposed by the compute module
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """Single captured evaluation frame."""

    P: Array2
    comps: Dict[str, Array2]
    E: float
    Gnorm: float
    stage_id: int
    iter_id: int


Event = Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]]


@dataclass
class RecorderViewPack:
    """Legacy result container used by :class:`cartoweave.compute.recorder.Recorder`."""

    frames: List[Frame]
    events: List[Event]
    last: Frame
    summary: Dict[str, Any]


# Energy callback signature: ``(P, scene, mask, cfg) -> (E, G, comps, meta)``
EnergyFn = Callable[
    [Array2, object, np.ndarray, dict],
    Tuple[float, Array2, Dict[str, Array2], dict],
]


def _grad_metrics(G: Optional[Array2]) -> Dict[str, float]:
    """Return infinity and L2 norms of ``G`` if provided."""

    if G is None:
        return {}
    return {
        "g_inf": float(np.linalg.norm(G, ord=np.inf)),
        "g_norm": float(np.linalg.norm(G)),
    }


__all__ = [
    "Array2",
    "EnergyFn",
    "Event",
    "Frame",
    "RecorderViewPack",
    "ViewPack",
    "VPFrame",
    "VPPass",
    "VPSources",
    "_grad_metrics",
]

