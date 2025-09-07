"""Pydantic models for compute and data configuration."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    weights: dict[str, float] = Field(default_factory=dict)
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

    model_config = ConfigDict(extra="forbid")


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
    modes: dict[
        str,
        Literal["midpoint", "centroid", "projected", "center", "nearest_edge"],
    ] = Field(default_factory=lambda: {"line": "midpoint", "area": "centroid"})

    model_config = ConfigDict(extra="forbid")


class DataTextFont(BaseModel):
    path: str
    size: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid")


class DataTextPadding(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid")


class DataTextGenerate(BaseModel):
    len_range: tuple[int, int]
    choose_lines: tuple[int, int]
    font: DataTextFont
    line_spacing_px: int = Field(ge=0)
    padding_px: DataTextPadding

    model_config = ConfigDict(extra="forbid")


class DataMutate(BaseModel):
    resample_text_on_size_mutate: bool = True

    model_config = ConfigDict(extra="forbid")


class DataGenerate(BaseModel):
    counts: DataCounts
    labels: int | None = Field(default=None, ge=0)
    label_mix: dict[str, float] | None = None
    frame_size: tuple[float, float]
    seed: int | None = None
    spacing: DataSpacing
    shapes: DataShapes
    route_gen: DataRouteGen
    area_gen: DataAreaGen
    anchors: DataAnchors
    text: DataTextGenerate
    mutate: DataMutate = Field(default_factory=DataMutate)
    save_path: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("frame_size")
    @classmethod
    def _check_frame(cls, v: tuple[float, float]) -> tuple[float, float]:
        if len(v) != 2 or any(not math.isfinite(x) or x <= 0 for x in v):
            raise ValueError("frame_size must be positive finite (W,H)")
        return v


class DataLoad(BaseModel):
    path: str

    model_config = ConfigDict(extra="forbid")


class DataConfig(BaseModel):
    source: Literal["generate", "load"] = "generate"
    generate: DataGenerate | None = None
    load: DataLoad | None = None
    action_num: int = Field(ge=0)

    model_config = ConfigDict(extra="allow")

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
        "DataTextFont",
        "DataTextPadding",
        "DataTextGenerate",
        "DataMutate",
        "DataGenerate",
        "DataLoad",
        "DataConfig",
        "ProfileData",
    ]
)
