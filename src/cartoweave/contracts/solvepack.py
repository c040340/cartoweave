from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Mapping
import numpy as np


def _fail(msg: str) -> None:
    raise ValueError(f"[SolvePack] {msg}")


def _as2(name: str, arr: Any, rows: int | None = None) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        _fail(f"{name} must have shape (N,2), got {a.shape}")
    if rows is not None and a.shape[0] != rows:
        _fail(f"{name} first dim must be {rows}, got {a.shape[0]}")
    if not np.isfinite(a).all():
        _fail(f"{name} contains non-finite values")
    return a


def _check_labels(labels: Any, L: int) -> None:
    if not isinstance(labels, list):
        _fail(f"labels must be a list, got {type(labels)}")
    if len(labels) != L:
        _fail(f"labels length {len(labels)} != L={L}")
    allowed = {"none", "point", "line", "area"}
    for i, lm in enumerate(labels):
        if not isinstance(lm, Mapping):
            _fail(f"labels[{i}] must be a mapping/dict, got {type(lm)}")
        kind = lm.get("anchor_kind", None)
        if kind not in allowed:
            _fail(f"labels[{i}].anchor_kind must be one of {allowed}, got {kind!r}")
        if not isinstance(lm.get("anchor_index", None), int):
            _fail(f"labels[{i}].anchor_index must be int")


def _check_lines(lines: Any) -> None:
    if lines is None:
        return
    if not isinstance(lines, list):
        _fail(f"lines must be a list of (K,2) arrays, got {type(lines)}")
    for j, Larr in enumerate(lines):
        _as2(f"lines[{j}]", Larr)


def _check_areas(areas: Any) -> None:
    if areas is None:
        return
    if not isinstance(areas, list):
        _fail(f"areas must be a list of (K,2) arrays, got {type(areas)}")
    for j, A in enumerate(areas):
        A = _as2(f"areas[{j}]", A)
        if A.shape[0] < 3:
            _fail(f"areas[{j}] must have at least 3 vertices, got {A.shape[0]}")


def _check_stages(stages: Any, L: int) -> None:
    if not isinstance(stages, list) or not stages:
        _fail("stages must be a non-empty list")
    for t, st in enumerate(stages):
        if not isinstance(st, Mapping):
            _fail(f"stages[{t}] must be a mapping/dict")
        iters = st.get("iters", None)
        if not isinstance(iters, int) or iters < 1:
            _fail(f"stages[{t}].iters must be int>=1, got {iters}")
        mask = st.get("mask_override", None)
        if mask is not None:
            if not isinstance(mask, np.ndarray) or mask.shape != (L,) or mask.dtype != bool:
                _fail(f"stages[{t}].mask_override must be bool array of shape ({L},)")


@dataclass(slots=True)
class SolvePack:
    # Core sizes
    L: int
    P0: np.ndarray                 # (L,2) float
    active_mask0: np.ndarray       # (L,) bool

    # Scene: all consumers in forces read keys from this dict
    scene: Dict[str, Any]          # must include keys below

    # Solver plumbing
    cfg: Dict[str, Any] = field(default_factory=dict)
    mode: str = "lbfgs"
    params: Dict[str, Any] = field(default_factory=dict)
    energy_and_grad: Optional[Any] = None  # callable or None
    stages: List[Dict[str, Any]] = field(default_factory=lambda: [{"iters": 6}])
    passes: List[str] = field(default_factory=lambda: ["schedule", "capture"])

    def __post_init__(self) -> None:
        # L / P0 / active_mask0
        if not isinstance(self.L, int) or self.L <= 0:
            _fail(f"L must be a positive int, got {self.L}")
        self.P0 = _as2("P0", self.P0, rows=self.L)
        if not (isinstance(self.active_mask0, np.ndarray) and
                self.active_mask0.shape == (self.L,) and
                self.active_mask0.dtype == bool):
            _fail(f"active_mask0 must be bool array of shape ({self.L},)")

        # scene required keys
        if not isinstance(self.scene, dict):
            _fail("scene must be a dict")
        required = ("frame_size", "labels", "WH", "points", "lines", "areas")
        for k in required:
            if k not in self.scene:
                _fail(f"scene missing required key: {k!r}")

        # frame_size
        fs = self.scene["frame_size"]
        if (not isinstance(fs, (tuple, list)) or len(fs) != 2 or
            not all(isinstance(v, (int, np.integer)) for v in fs) or
            fs[0] <= 0 or fs[1] <= 0):
            _fail(f"frame_size must be (W>0, H>0) ints, got {fs!r}")

        # labels & WH
        _check_labels(self.scene["labels"], self.L)
        self.scene["WH"] = _as2("WH", self.scene["WH"], rows=self.L)
        if (np.asarray(self.scene["WH"]) < 0).any():
            _fail("WH must be non-negative")

        # points / lines / areas
        _as2("points", self.scene["points"])
        _check_lines(self.scene["lines"])
        _check_areas(self.scene["areas"])

        # stages
        _check_stages(self.stages, self.L)

        # --- anchor index range checks (recommended) ---
        labels = self.scene["labels"]
        pts    = np.asarray(self.scene["points"])
        lines  = self.scene["lines"]
        areas  = self.scene["areas"]

        for i, lm in enumerate(labels):
            kind = lm.get("anchor_kind")
            idx  = lm.get("anchor_index")
            if kind == "point":
                if not (0 <= idx < len(pts)):
                    _fail(f"labels[{i}].anchor_index out of range for points: {idx} not in [0,{len(pts)})")
            elif kind == "line":
                if not (0 <= idx < len(lines)):
                    _fail(f"labels[{i}].anchor_index out of range for lines: {idx} not in [0,{len(lines)})")
            elif kind == "area":
                if not (0 <= idx < len(areas)):
                    _fail(f"labels[{i}].anchor_index out of range for areas: {idx} not in [0,{len(areas)})")
            elif kind == "none":
                # unanchored label; no index check
                pass

    def validate(self) -> None:
        """Backward-compatible no-op; validation happens in __post_init__."""
        return None
