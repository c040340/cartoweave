"""SolvePack v2 strict contract using Pydantic models.

This module defines the "SolvePack" schema used by the compute layer.  The
contract is intentionally strict and non‑legacy: geometry is stored only in the
scene pools and labels carry no embedded geometry.  Any unknown fields result in
validation errors.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Helper type aliases
# ---------------------------------------------------------------------------

XY = tuple[float, float]


# ---------------------------------------------------------------------------
# Scene model
# ---------------------------------------------------------------------------

class Scene(BaseModel):
    """Static geometry containers of the scene."""

    frame_size: XY
    points: list[XY] = Field(default_factory=list)
    lines: list[list[XY]] = Field(default_factory=list)
    areas: list[list[XY]] = Field(default_factory=list)
    bounds: tuple[float, float, float, float] | None = None
    padding: float | None = None

    model_config = ConfigDict(extra="forbid")

    # --- validators -----------------------------------------------------
    @field_validator("frame_size")
    @classmethod
    def _check_frame_size(cls, v: XY) -> XY:
        if len(v) != 2 or any(not math.isfinite(x) or x <= 0 for x in v):
            raise ValueError("frame_size must be positive finite (W,H)")
        return v

    @field_validator("points")
    @classmethod
    def _check_points(cls, v: list[XY]) -> list[XY]:
        for i, xy in enumerate(v):
            if len(xy) != 2 or any(not math.isfinite(c) for c in xy):
                raise ValueError(f"points[{i}] must be finite (x,y)")
        return v

    @field_validator("lines")
    @classmethod
    def _check_lines(cls, v: list[list[XY]]) -> list[list[XY]]:
        for i, line in enumerate(v):
            if len(line) < 2:
                raise ValueError(f"lines[{i}] must have >=2 vertices")
            for j, xy in enumerate(line):
                if len(xy) != 2 or any(not math.isfinite(c) for c in xy):
                    raise ValueError(f"lines[{i}][{j}] must be finite (x,y)")
        return v

    @field_validator("areas")
    @classmethod
    def _check_areas(cls, v: list[list[XY]]) -> list[list[XY]]:
        for i, area in enumerate(v):
            if len(area) < 3:
                raise ValueError(f"areas[{i}] must have >=3 vertices")
            for j, xy in enumerate(area):
                if len(xy) != 2 or any(not math.isfinite(c) for c in xy):
                    raise ValueError(f"areas[{i}][{j}] must be finite (x,y)")
        return v

    @field_validator("bounds")
    @classmethod
    def _check_bounds(cls, v: tuple[float, float, float, float] | None):
        if v is None:
            return v
        if len(v) != 4 or any(not math.isfinite(x) for x in v):
            raise ValueError("bounds must be finite (xmin,ymin,xmax,ymax)")
        return v

    @field_validator("padding")
    @classmethod
    def _check_padding(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not math.isfinite(v) or v < 0:
            raise ValueError("padding must be >=0 and finite")
        return v


# ---------------------------------------------------------------------------
# Anchor model
# ---------------------------------------------------------------------------

AnchorTarget = Literal["free", "point", "line", "area"]
PointMode = Literal["exact", "offset"]
LineMode = Literal["midpoint", "projected", "centroid"]
AreaMode = Literal["centroid", "center", "nearest_edge"]
FreeMode = Literal["xy"]


class Anchor(BaseModel):
    target: AnchorTarget
    index: int | None = None
    mode: PointMode | LineMode | AreaMode | FreeMode
    xy: XY | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check(self):
        if self.target == "free":
            if self.index is not None:
                raise ValueError("index must be None when target='free'")
            if self.mode != "xy":
                raise ValueError("mode must be 'xy' when target='free'")
            if self.xy is None:
                raise ValueError("xy required when target='free'")
            if len(self.xy) != 2 or any(not math.isfinite(c) for c in self.xy):
                raise ValueError("xy must be finite (x,y)")
        else:
            if self.index is None or self.index < 0:
                raise ValueError("index must be >=0 when target!='free'")
            if self.target == "point":
                if self.mode not in ("exact", "offset"):
                    raise ValueError("invalid mode for point target")
                if self.mode == "offset":
                    if self.xy is not None:
                        if len(self.xy) != 2 or any(not math.isfinite(c) for c in self.xy):
                            raise ValueError("xy must be finite (x,y)")
                elif self.xy is not None:
                    raise ValueError("xy only allowed for point/offset")
            elif self.target == "line":
                if self.mode not in ("midpoint", "projected", "centroid"):
                    raise ValueError("invalid mode for line target")
                if self.xy is not None:
                    raise ValueError("xy only allowed for point/offset")
            elif self.target == "area":
                if self.mode not in ("centroid", "center", "nearest_edge"):
                    raise ValueError("invalid mode for area target")
                if self.xy is not None:
                    raise ValueError("xy only allowed for point/offset")
        return self


# ---------------------------------------------------------------------------
# Label model
# ---------------------------------------------------------------------------

LabelKind = Literal["point", "line", "area"]


class Label(BaseModel):
    id: int
    kind: LabelKind
    WH: XY | None = None
    thickness: float | None = None
    radius: float | None = None
    anchor: Anchor
    ll_kernel: Literal["rect", "disk"] | None = "rect"
    mode: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _no_geometry(cls, data: Any):
        if isinstance(data, dict) and ("polyline" in data or "polygon" in data):
            raise ValueError(
                "Label must not carry geometry; put lines/areas in scene0.* and reference via anchor."
            )
        return data

    @field_validator("WH")
    @classmethod
    def _check_wh(cls, v: XY | None) -> XY | None:
        if v is None:
            return v
        if len(v) != 2 or any(not math.isfinite(x) or x <= 0 for x in v):
            raise ValueError("WH must be positive finite (W,H)")
        return v

    @field_validator("thickness", "radius")
    @classmethod
    def _check_positive(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not math.isfinite(v) or v <= 0:
            raise ValueError("values must be positive and finite")
        return v


# ---------------------------------------------------------------------------
# ActionRecord model
# ---------------------------------------------------------------------------


class ActionRecord(BaseModel):
    t: float
    step: int | None = None
    id: int
    type: Literal["appear", "mutate", "disappear"]
    kind_to: str | None = None
    WH_to: XY | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("t")
    @classmethod
    def _check_t(cls, v: float) -> float:
        if not (0.0 < float(v) < 1.0):
            raise ValueError("t must be in (0,1)")
        return float(v)


# ---------------------------------------------------------------------------
# SolvePack model
# ---------------------------------------------------------------------------


class SolvePack(BaseModel):
    L: int
    P0: list[XY]
    labels0: list[Label]
    active0: list[StrictBool]
    scene0: Scene
    cfg: dict[str, Any]
    rng_seed: int | None = None
    uid: str | None = None
    created_at: str | None = None
    actions: list[ActionRecord] = Field(default_factory=list)
    action_num: int | None = None
    behaviors: list[dict] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("P0")
    @classmethod
    def _check_p0(cls, v: list[XY]) -> list[XY]:
        for i, xy in enumerate(v):
            if len(xy) != 2 or any(not math.isfinite(c) for c in xy):
                raise ValueError(f"P0[{i}] must be finite (x,y), got {xy}")
        return v

    @field_validator("active0")
    @classmethod
    def _check_active(cls, v: list[StrictBool]) -> list[StrictBool]:
        for i, b in enumerate(v):
            if not isinstance(b, bool):
                raise ValueError(f"active0[{i}] must be bool")
        return v

    @model_validator(mode="after")
    def _check_actions(self):
        times = [a.t for a in self.actions]
        if any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("actions times must be strictly increasing")
        if self.action_num is not None and len(self.actions) != self.action_num:
            raise ValueError("len(actions) must equal action_num")
        return self

    @model_validator(mode="after")
    def _check_lengths(self):
        if self.L < 0:
            raise ValueError(f"L must be >=0, got {self.L}")
        if not (len(self.P0) == len(self.labels0) == len(self.active0) == self.L):
            raise ValueError(
                f"length mismatch: L={self.L} P0={len(self.P0)} "
                f"labels0={len(self.labels0)} active0={len(self.active0)}"
            )
        for i, lbl in enumerate(self.labels0):
            if lbl.id != i:
                raise ValueError(f"labels0[{i}].id expected {i}, got {lbl.id}")
        # anchor reference range checks
        for lbl in self.labels0:
            a = lbl.anchor
            if a.target == "point" and a.index >= len(self.scene0.points):
                raise ValueError("anchor index out of range for points pool")
            if a.target == "line" and a.index >= len(self.scene0.lines):
                raise ValueError("anchor index out of range for lines pool")
            if a.target == "area" and a.index >= len(self.scene0.areas):
                raise ValueError("anchor index out of range for areas pool")
        return self

    @model_validator(mode="after")
    def _check_cfg(self):
        keys = set(self.cfg.keys())
        if keys != {"compute"}:
            raise ValueError(f"cfg top-level keys must be {{'compute'}}, got {keys}")
        return self

    def validate(self) -> SolvePack:  # pragma: no cover - compatibility helper
        """Explicit validation; simply returns ``self`` on success."""
        return self


