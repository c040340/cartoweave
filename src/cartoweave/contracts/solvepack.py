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
    """Validate shapes and basic integrity of a :class:`SolvePack` (v2)."""

    # --------------------------
    # base shapes (you already had)
    # --------------------------
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

    # --------------------------
    # scene checks (be stricter)
    # --------------------------
    sc = pack.scene0
    if not isinstance(sc, Scene):
        raise ValueError("scene0 must be Scene")

    pts = np.asarray(sc.points, float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("scene.points must be (M,2)")
    if not np.isfinite(pts).all():
        raise ValueError("scene.points contain non-finite")

    # lines: each (m>=2, 2)
    if not isinstance(sc.lines, list):
        raise ValueError("scene.lines must be a list")
    for j, ln in enumerate(sc.lines):
        arr = np.asarray(ln, float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
            raise ValueError(f"scene.lines[{j}] must be (m>=2,2), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"scene.lines[{j}] contains non-finite")

    # areas: each dict with 'exterior' (k>=3,2); holes optional
    if not isinstance(sc.areas, list):
        raise ValueError("scene.areas must be a list")
    for j, ar in enumerate(sc.areas):
        if not isinstance(ar, dict) or "exterior" not in ar:
            raise ValueError(f"scene.areas[{j}] must be dict with 'exterior'")
        ext = np.asarray(ar["exterior"], float)
        if ext.ndim != 2 or ext.shape[1] != 2 or ext.shape[0] < 3:
            raise ValueError(f"scene.areas[{j}]['exterior'] must be (k>=3,2), got {ext.shape}")
        if not np.isfinite(ext).all():
            raise ValueError(f"scene.areas[{j}]['exterior'] contains non-finite")
        holes = ar.get("holes", [])
        if holes is not None:
            if not isinstance(holes, list):
                raise ValueError(f"scene.areas[{j}]['holes'] must be list or None")
            for hidx, h in enumerate(holes):
                h = np.asarray(h, float)
                if h.ndim != 2 or h.shape[1] != 2 or h.shape[0] < 3:
                    raise ValueError(f"scene.areas[{j}]['holes'][{hidx}] must be (k>=3,2)")
                if not np.isfinite(h).all():
                    raise ValueError(f"scene.areas[{j}]['holes'][{hidx}] contains non-finite")

    # frame_size
    fs = tuple(sc.frame_size)
    if len(fs) != 2 or not all(np.isfinite(fs)) or fs[0] <= 0 or fs[1] <= 0:
        raise ValueError(f"scene.frame_size must be positive 2-tuple, got {sc.frame_size}")

    # optional scene.WH
    if sc.WH.shape not in [(0, 2), (N, 2)]:
        raise ValueError("scene.WH must be (N,2) or empty")
    if sc.WH.size and (sc.WH < 0).any():
        raise ValueError("scene.WH must be non-negative when provided")

    counts = (len(pts), len(sc.lines), len(sc.areas))  # for anchor bounds

    # --------------------------
    # labels0 checks
    # --------------------------
    ALLOWED_KINDS = {"none", "point", "line", "area"}

    for i, lbl in enumerate(pack.labels0):
        if not isinstance(lbl, LabelState):
            raise ValueError(f"labels0[{i}] must be LabelState")
        # kind
        if lbl.kind not in ALLOWED_KINDS:
            raise ValueError(f"labels0[{i}].kind must be one of {ALLOWED_KINDS}, got {lbl.kind!r}")
        # WH
        wh = np.asarray(lbl.WH, float)
        if wh.shape != (2,):
            raise ValueError(f"labels0[{i}].WH must have shape (2,), got {wh.shape}")
        if not np.isfinite(wh).all() or (wh < 0).any():
            raise ValueError(f"labels0[{i}].WH must be finite and non-negative")
        # anchor (optional)
        if lbl.anchor is not None:
            ak = getattr(lbl.anchor, "kind", None)
            ai = getattr(lbl.anchor, "index", None)
            if ak not in {"point", "line", "area"}:
                raise ValueError(f"labels0[{i}].anchor.kind invalid: {ak!r}")
            if not isinstance(ai, int) or ai < 0:
                raise ValueError(f"labels0[{i}].anchor.index must be non-negative int")
            if ak == "point" and ai >= counts[0]: raise ValueError(f"labels0[{i}].anchor.index out of range for points")
            if ak == "line"  and ai >= counts[1]: raise ValueError(f"labels0[{i}].anchor.index out of range for lines")
            if ak == "area"  and ai >= counts[2]: raise ValueError(f"labels0[{i}].anchor.index out of range for areas")

    # --------------------------
    # cfg / behaviors checks
    # --------------------------
    if not isinstance(pack.cfg, dict):
        raise ValueError("cfg must be a dict")

    # behavior policy defaults
    behavior_cfg = pack.cfg.get("behavior", {})
    if not isinstance(behavior_cfg, dict):
        raise ValueError("cfg['behavior'] must be a dict")
    for k in ["place_on_first_activation", "snap_on_kind_change"]:
        if k in behavior_cfg and not isinstance(behavior_cfg[k], bool):
            raise ValueError(f"cfg['behavior']['{k}'] must be bool")

    # default_WH
    dwh = behavior_cfg.get("default_WH", {"point": [8, 8], "line": [12, 6], "area": [40, 30]})
    if not isinstance(dwh, dict) or not all(t in dwh for t in ("point","line","area")):
        raise ValueError("cfg['behavior']['default_WH'] must have point/line/area")
    for t in ("point","line","area"):
        arr = np.asarray(dwh[t], float)
        if arr.shape != (2,) or (arr < 0).any() or not np.isfinite(arr).all():
            raise ValueError(f"cfg['behavior']['default_WH']['{t}'] must be length-2 non-negative")

    # behaviors list
    behs = pack.cfg.get("behaviors")
    if not isinstance(behs, list):
        raise ValueError("cfg['behaviors'] must be a list")
    # ops must be dict-like with lists inside
    for si, beh in enumerate(behs):
        # Behavior dataclass or dict-equivalent (your v2 uses dict ops)
        iters = getattr(beh, "iters", None)
        solver = getattr(beh, "solver", None)
        ops = getattr(beh, "ops", None)
        params = getattr(beh, "params", {})

        if not isinstance(iters, int) or iters < 1:
            raise ValueError(f"behaviors[{si}].iters must be positive int")
        if not isinstance(solver, str):
            raise ValueError(f"behaviors[{si}].solver must be str")
        if not isinstance(ops, dict):
            raise ValueError(f"behaviors[{si}].ops must be dict")

        act = ops.get("activate", [])
        deact = ops.get("deactivate", [])
        mut = ops.get("mutate", [])

        if not isinstance(act, list) or not isinstance(deact, list) or not isinstance(mut, list):
            raise ValueError(f"behaviors[{si}].ops.* must be lists")

        # id bounds
        for lid in list(act) + list(deact):
            if not isinstance(lid, int) or lid < 0 or lid >= N:
                raise ValueError(f"behaviors[{si}].ops id out of range: {lid}")

        # mutate entries
        for mi, m in enumerate(mut):
            if not isinstance(m, dict):
                raise ValueError(f"behaviors[{si}].ops.mutate[{mi}] must be dict")
            if "id" not in m or "set" not in m:
                raise ValueError(f"behaviors[{si}].ops.mutate[{mi}] must contain 'id' and 'set'")
            lid = m["id"]
            if not isinstance(lid, int) or lid < 0 or lid >= N:
                raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].id out of range")
            setv = m["set"]
            if not isinstance(setv, dict):
                raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].set must be dict")

            # optional kind
            if "kind" in setv and setv["kind"] not in ALLOWED_KINDS:
                raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].set.kind invalid")

            # optional WH
            if "WH" in setv:
                arr = np.asarray(setv["WH"], float)
                if arr.shape != (2,) or (arr < 0).any() or not np.isfinite(arr).all():
                    raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].set.WH must be length-2 non-negative")

            # optional anchor
            if "anchor" in setv:
                anc = setv["anchor"]
                if not isinstance(anc, dict) or "kind" not in anc or "index" not in anc:
                    raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].set.anchor must have kind/index")
                ak, ai = anc["kind"], anc["index"]
                if ak not in {"point","line","area"}:
                    raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].set.anchor.kind invalid")
                if not isinstance(ai, int) or ai < 0:
                    raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].set.anchor.index must be non-negative int")
                if ak == "point" and ai >= counts[0]: raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].set.anchor.index out of range for points")
                if ak == "line"  and ai >= counts[1]: raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].set.anchor.index out of range for lines")
                if ak == "area"  and ai >= counts[2]: raise ValueError(f"behaviors[{si}].ops.mutate[{mi}].set.anchor.index out of range for areas")

    return pack

