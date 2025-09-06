"""Strict SolvePack schema using Pydantic models.

This module defines a minimal contract used by the compute layer. The schema is
not backwards compatible with the previous dataclass based implementation. Any
unknown fields will raise validation errors.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

# ---------------------------------------------------------------------------
# Helper type aliases
# ---------------------------------------------------------------------------

XY = tuple[float, float]
Polygon = list[XY]
Polyline = list[XY]


# ---------------------------------------------------------------------------
# Scene model
# ---------------------------------------------------------------------------

class Scene(BaseModel):
    """Static geometry of the scene."""

    frame_size: XY
    bounds: tuple[float, float, float, float] | None = None
    padding: float | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("frame_size")
    @classmethod
    def _check_frame_size(cls, v: XY) -> XY:
        if len(v) != 2 or any(not math.isfinite(x) or x <= 0 for x in v):
            raise ValueError("frame_size must be positive finite (W,H)")
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
# Label model
# ---------------------------------------------------------------------------

class Anchor(BaseModel):
    mode: Literal["xy", "centroid", "line_midpoint", "bbox_center"]
    xy: XY | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_xy(self):  # type: ignore[override]
        if self.mode == "xy":
            if self.xy is None:
                raise ValueError("anchor.xy required when mode='xy'")
            if len(self.xy) != 2 or any(not math.isfinite(x) for x in self.xy):
                raise ValueError("anchor.xy must be finite (x,y)")
        elif self.xy is not None:
            raise ValueError("anchor.xy only allowed when mode='xy'")
        return self


class Label(BaseModel):
    id: int
    kind: Literal["point", "line", "area"]
    WH: XY | None = None
    thickness: float | None = None
    radius: float | None = None
    polyline: Polyline | None = None
    polygon: Polygon | None = None
    anchor: Anchor
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

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

    @field_validator("polyline", "polygon")
    @classmethod
    def _check_poly(cls, v: Polyline | Polygon | None, info):
        if v is None:
            return v
        for xy in v:
            if len(xy) != 2 or any(not math.isfinite(c) for c in xy):
                raise ValueError(f"{info.field_name} must contain finite (x,y) pairs")
        return v

    @model_validator(mode="after")
    def _kind_requirements(self):  # type: ignore[override]
        if self.kind == "point":
            if self.polyline is not None or self.polygon is not None:
                raise ValueError("point label must not have polyline/polygon")
        elif self.kind == "line":
            if not self.polyline or len(self.polyline) < 2:
                raise ValueError("line label requires polyline with >=2 vertices")
            if self.polygon is not None:
                raise ValueError("line label must not have polygon")
        elif self.kind == "area":
            if not self.polygon or len(self.polygon) < 3:
                raise ValueError("area label requires polygon with >=3 vertices")
            if self.polyline is not None:
                raise ValueError("area label must not have polyline")
        return self


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

    model_config = ConfigDict(extra="forbid")

    @field_validator("P0")
    @classmethod
    def _check_p0(cls, v: list[XY]) -> list[XY]:
        for i, xy in enumerate(v):
            if len(xy) != 2 or any(not math.isfinite(c) for c in xy):
                raise ValueError(f"P0[{i}] must be finite (x,y), got {xy}")
        return v

    @model_validator(mode="after")
    def _check_lengths(self):  # type: ignore[override]
        if self.L <= 0:
            raise ValueError(f"L must be positive, got {self.L}")
        if not (len(self.P0) == len(self.labels0) == len(self.active0) == self.L):
            raise ValueError(
                f"length mismatch: L={self.L} P0={len(self.P0)} "
                f"labels0={len(self.labels0)} active0={len(self.active0)}"
            )
        for i, lbl in enumerate(self.labels0):
            if lbl.id != i:
                raise ValueError(
                    f"labels0[{i}].id expected {i}, got {lbl.id}"
                )
        return self

    @model_validator(mode="after")
    def _check_cfg(self):  # type: ignore[override]
        if "solver" in self.cfg:
            raise ValueError("cfg contains legacy key 'solver'")
        if "terms" in self.cfg:
            raise ValueError("cfg contains legacy key 'terms'")
        keys = set(self.cfg.keys())
        if keys != {"compute"}:
            raise ValueError(f"cfg top-level keys must be {{'compute'}}, got {keys}")
        return self

    def validate(self) -> SolvePack:  # pragma: no cover - compatibility helper
        """Explicit validation; simply returns ``self`` on success."""
        return self


__all__ = ["SolvePack", "Scene", "Label", "Anchor"]
