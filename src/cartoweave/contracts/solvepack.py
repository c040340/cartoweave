"""Dataclasses defining the SolvePack v2 contract.

This module intentionally keeps the structures light weight and Pythonic so
that callers can assemble packs either from generated content or loaded
snapshots.  The compute layer largely operates on dictionaries, therefore the
objects below expose small helpers such as ``get`` so that they behave similarly
to mappings when consumed by legacy code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

import numpy as np

__all__ = [
    "Kind",
    "AnchorKind",
    "AnchorSpec",
    "LabelState",
    "BehaviorOp",
    "Behavior",
    "Scene",
    "SolvePack",
    "validate",
]

# basic string enums -----------------------------------------------------------

# ``Kind`` represents the label kind used by the compute layer.  ``"none"`` is
# a valid placeholder for inactive labels.
Kind = Literal["none", "point", "line", "area"]

# Anchors are restricted to geometry primitives and do not include ``"none"``.
AnchorKind = Literal["point", "line", "area"]


# ---------------------------------------------------------------------------
#  Basic dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnchorSpec:
    kind: str
    index: int
    t: Optional[float] = None


@dataclass(slots=True)
class LabelState:
    kind: str
    WH: np.ndarray
    anchor: AnchorSpec | None = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BehaviorOp:
    id: int
    set: Dict[str, Any]


@dataclass(slots=True)
class Behavior:
    iters: int
    ops: Dict[str, Any]
    solver: str = "lbfgs"
    params: Dict[str, Any] = field(default_factory=dict)

    # ``BehaviorPass`` calls ``beh.get(...)`` on these objects.  Implement a
    # tiny helper so dataclasses behave like dictionaries.
    def get(self, key: str, default: Any | None = None) -> Any:
        return getattr(self, key, default)


@dataclass(slots=True)
class Scene:
    points: np.ndarray
    lines: List[np.ndarray]
    areas: List[Dict[str, Any]]
    frame_size: tuple[float, float]
    labels: List[Dict[str, Any]] = field(default_factory=list)
    WH: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    # ``BehaviorPass`` accesses scene like a mapping.  ``to_dict`` plus ``get``
    # keep compatibility while retaining the dataclass definition.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": self.points,
            "lines": self.lines,
            "areas": self.areas,
            "frame_size": self.frame_size,
            "labels": self.labels,
            "WH": self.WH,
        }

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.to_dict().get(key, default)


# ---------------------------------------------------------------------------
#  SolvePack and validation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SolvePack:
    N: int
    P0: np.ndarray
    active0: np.ndarray
    labels0: List[LabelState]
    scene0: Scene
    cfg: Dict[str, Any] = field(default_factory=dict)

    # -- compatibility helpers -------------------------------------------------
    @property
    def L(self) -> int:  # pragma: no cover - legacy alias
        return self.N

    @property
    def active_mask0(self) -> np.ndarray:  # pragma: no cover - legacy alias
        return self.active0

    def validate(self) -> None:  # pragma: no cover - user convenience
        validate(self)


def validate(pack: SolvePack) -> SolvePack:
    """Validate shapes and basic integrity of a :class:`SolvePack`.

    The checks here are intentionally lightweight; they aim to catch gross shape
    mismatches while staying cheap enough to run in examples.
    """

    N = pack.N
    if not isinstance(N, int) or N < 1:
        raise ValueError(f"N must be a positive int, got {N!r}")

    P0 = np.asarray(pack.P0, float)
    if P0.shape != (N, 2):
        raise ValueError(f"P0 must have shape (N,2), got {P0.shape}")
    if not np.isfinite(P0).all():
        raise ValueError("P0 contains non-finite values")

    active0 = np.asarray(pack.active0, bool)
    if active0.shape != (N,):
        raise ValueError(f"active0 must have shape (N,), got {active0.shape}")

    if len(pack.labels0) != N:
        raise ValueError("labels0 length mismatch")
    for i, lbl in enumerate(pack.labels0):
        if not isinstance(lbl, LabelState):
            raise ValueError(f"labels0[{i}] must be LabelState")
        if lbl.WH.shape != (2,):
            raise ValueError(f"labels0[{i}].WH must have shape (2,)")

    sc = pack.scene0
    if not isinstance(sc, Scene):
        raise ValueError("scene0 must be Scene")
    pts = np.asarray(sc.points, float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("scene.points must be (M,2)")
    if sc.WH.shape not in [(0, 2), (N, 2)]:
        raise ValueError("scene.WH must be (N,2) or empty")

    return pack

