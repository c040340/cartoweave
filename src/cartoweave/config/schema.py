"""Pydantic models for compute and data configuration."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class Eps(BaseModel):
    div: float = Field(gt=0)
    sqrt: float = Field(gt=0)
    proj: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class CapturePass(BaseModel):
    every: int = Field(ge=1)
    final_always: bool

    model_config = ConfigDict(extra="forbid")


class GradClipPass(BaseModel):
    enable: bool
    norm_max: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class StepLimitPass(BaseModel):
    max_step_norm: float | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class Passes(BaseModel):
    capture: CapturePass
    grad_clip: GradClipPass
    step_limit: StepLimitPass

    model_config = ConfigDict(extra="forbid")


class SolverPublic(BaseModel):
    mode: str
    use_warmup: bool

    model_config = ConfigDict(extra="forbid")


class SolverTuningLBFGSB(BaseModel):
    m: int = Field(ge=1)
    maxiter: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid")


class SolverTuningStopping(BaseModel):
    gtol: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class SolverTuningAntiJump(BaseModel):
    step_cap_px: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class SolverTuningWarmup(BaseModel):
    steps: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid")


class SolverTuning(BaseModel):
    lbfgsb: SolverTuningLBFGSB
    stopping: SolverTuningStopping
    anti_jump: SolverTuningAntiJump
    warmup: SolverTuningWarmup

    model_config = ConfigDict(extra="forbid")


class SolverInternalsClip(BaseModel):
    force_abs_max: float = Field(gt=0)
    energy_abs_max: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class SolverInternalsStability(BaseModel):
    exp_clip: float
    eps_norm: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class SolverInternals(BaseModel):
    clip: SolverInternalsClip
    stability: SolverInternalsStability

    model_config = ConfigDict(extra="forbid")


class Solver(BaseModel):
    public: SolverPublic
    tuning: SolverTuning
    internals: SolverInternals

    model_config = ConfigDict(extra="forbid")


class Compute(BaseModel):
    eps: Eps
    passes: Passes
    weights: Dict[str, float] = Field(default_factory=dict)
    solver: Solver

    model_config = ConfigDict(extra="forbid")

__all__ = ["Compute"]


# ---------------------------------------------------------------------------
# Data configuration models
# ---------------------------------------------------------------------------


class DataCounts(BaseModel):
    points: int = Field(ge=0)
    lines: int = Field(ge=0)
    areas: int = Field(ge=0)
    total_labels: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_total(self):  # type: ignore[override]
        if self.total_labels is not None:
            if self.total_labels != self.points + self.lines + self.areas:
                raise ValueError("total_labels must equal points+lines+areas")
        return self


class DataSpacing(BaseModel):
    min_point_dist: float = Field(ge=0.0)
    margin: float = Field(ge=0.0)

    model_config = ConfigDict(extra="forbid")


class DataShapes(BaseModel):
    area_fallback: Literal["polygon", "capsule"] = "polygon"
    line_min_vertices: int = Field(default=2, ge=2)

    model_config = ConfigDict(extra="forbid")


class DataRouteGen(BaseModel):
    min_vertex_spacing_scale: float = Field(default=1.0, ge=0.0)
    inset_margin_scale: float = Field(default=0.05, ge=0.0)
    segment_len_scale: float = Field(default=0.10, ge=0.0)
    angle_noise_sigma: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class DataAreaGen(BaseModel):
    inset_margin_scale: float = Field(default=0.04, ge=0.0)
    min_edge_spacing_scale: float = Field(default=0.04, ge=0.0)
    n_vertices_min: int = Field(default=3, ge=3)
    n_vertices_max: int = Field(default=7, ge=3)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_vertices(self):  # type: ignore[override]
        if self.n_vertices_max < self.n_vertices_min:
            raise ValueError("n_vertices_max must be >= n_vertices_min")
        return self


class DataAnchors(BaseModel):
    policy: Literal["round_robin", "random", "fixed"] = "round_robin"
    modes: Dict[
        str,
        Literal["midpoint", "centroid", "projected", "center", "nearest_edge"],
    ] = Field(default_factory=lambda: {"line": "midpoint", "area": "centroid"})

    model_config = ConfigDict(extra="forbid")


class DataGenerate(BaseModel):
    counts: DataCounts
    steps: int = Field(ge=1)
    frame_size: Tuple[float, float]
    seed: Optional[int] = None
    spacing: DataSpacing
    shapes: DataShapes
    route_gen: DataRouteGen
    area_gen: DataAreaGen
    anchors: DataAnchors

    model_config = ConfigDict(extra="forbid")

    @field_validator("frame_size")
    @classmethod
    def _check_frame(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if len(v) != 2 or any(not math.isfinite(x) or x <= 0 for x in v):
            raise ValueError("frame_size must be positive finite (W,H)")
        return v


class DataLoad(BaseModel):
    path: str

    model_config = ConfigDict(extra="forbid")


class DataConfig(BaseModel):
    source: Literal["generate", "load"] = "generate"
    generate: Optional[DataGenerate] = None
    load: Optional[DataLoad] = None
    behaviors: List[dict] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_source(self):  # type: ignore[override]
        if self.source == "generate":
            if self.generate is None:
                raise ValueError("generate config required when source='generate'")
        elif self.source == "load":
            if self.load is None:
                raise ValueError("load config required when source='load'")
        else:  # pragma: no cover
            raise ValueError(f"unknown source: {self.source}")
        return self


class ProfileData(BaseModel):
    data: DataConfig

    model_config = ConfigDict(extra="forbid")


__all__.extend(
    [
        "DataCounts",
        "DataSpacing",
        "DataShapes",
        "DataRouteGen",
        "DataAreaGen",
        "DataAnchors",
        "DataGenerate",
        "DataLoad",
        "DataConfig",
        "ProfileData",
    ]
)
