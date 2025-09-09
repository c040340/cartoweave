"""Pydantic models for compute and data configuration."""
from __future__ import annotations

import math
from typing import Dict, Literal, List

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator


class Eps(BaseModel):
    numeric: float = Field(gt=0)
    dist: float = Field(gt=0)
    abs: float = Field(gt=0)
    proj: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class CapturePass(BaseModel):
    every: int = Field(ge=1)
    final_always: bool
    limit: int | None = Field(default=None, ge=1)
    mode: Literal["none", "warmup", "main", "both"] = "both"

    model_config = ConfigDict(extra="forbid")


class GradClipPass(BaseModel):
    enable: bool
    max_norm: float | None = Field(default=None, gt=0)
    max_abs: float | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class StepLimitPass(BaseModel):
    max_step_norm: float | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class ActionPass(BaseModel):
    enable: bool = True

    model_config = ConfigDict(extra="forbid")


class GeomPreprocPass(BaseModel):
    enable: bool = True
    tiny_eps: float = Field(default=1e-9, gt=0)

    model_config = ConfigDict(extra="forbid")


class NanGuardPass(BaseModel):
    enable: bool = True
    on_nan: str = "zero"
    on_inf: str = "clip"
    e_fallback: float = 0.0

    model_config = ConfigDict(extra="forbid")


class CalibrationPass(BaseModel):
    enable: bool = False
    clip_q: float = Field(default=0.995, gt=0, lt=1)
    p_q: float = Field(default=0.95, gt=0, lt=1)
    clamp_min: float = Field(default=0.33, gt=0)
    clamp_max: float = Field(default=3.0, gt=0)
    min_act: float = Field(default=0.05, ge=0, le=1)
    hysteresis: float = Field(default=0.25, ge=0)
    ema_alpha: float = Field(default=1.0, ge=0, le=1)
    base_term: str = "focus.attract"
    target_rel: Dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class Passes(BaseModel):
    capture: CapturePass
    grad_clip: GradClipPass
    step_limit: StepLimitPass
    nan_guard: NanGuardPass
    calibration: CalibrationPass
    action: ActionPass = ActionPass()
    geom_preproc: GeomPreprocPass = GeomPreprocPass()

    model_config = ConfigDict(extra="forbid")


class Kernel(BaseModel):
    model: Literal["inv_pow", "poly", "exp", "logcosh"] = "inv_pow"
    exponent: float = Field(default=2.0, gt=0)
    soft_eps: float = Field(default=1e-6, ge=0)
    gate_gamma: float | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class BetaParams(BaseModel):
    sep: float | None = Field(default=None, gt=0)
    in_: float | None = Field(default=None, alias="in", gt=0)
    dist: float | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class EpsTerm(BaseModel):
    numeric: float | None = Field(default=None, gt=0)
    dist: float | None = Field(default=None, ge=0)
    abs: float | None = Field(default=None, ge=0)
    proj: float | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class TanGate(BaseModel):
    eta: float = Field(gt=0)
    slack: float = Field(ge=0)

    model_config = ConfigDict(extra="forbid")


class TermParams(BaseModel):
    enable: bool = False
    k_out: float | None = None
    k_in: float | None = None
    k_wall: float | None = None
    k_push: float | None = None
    k_embed: float | None = None
    k_tan: float | None = None
    k_cross: float | None = None
    k_spring: float | None = None
    k_attract: float | None = None
    intro_times: float | None = None
    kernel: Kernel | None = None
    beta: BetaParams | float | None = None
    beta_edge: float | None = Field(default=None, gt=0)
    g_eps: float | None = Field(default=None, ge=0)
    eps: EpsTerm | None = None
    min_gap: float | None = None
    alpha: float | None = None
    eta: float | None = None
    tan_cap_scale: float | None = None
    gate_min_interior: float | None = None
    kappa: float | None = None
    beta_smax: float | None = None
    use_logcosh: bool | None = None
    sat_p0: float | None = None
    outside_weight: float | None = None
    in_decay: float | None = None
    zero_dist: float | None = None
    out_decay: float | None = None
    ratio_in: float | None = None
    tan_gate: TanGate | None = None
    pad: float | None = None
    y_down: bool | None = None
    center: List[float] | None = None
    sigma: float | None = None
    wh: List[float] | None = None
    delta: float | None = None
    only_free: bool | None = None
    use_legacy_gate: bool | None = None
    mode: str | None = None
    r0: float | None = None
    huber_delta: float | None = None
    rect_alpha: float | None = None
    smooth_abs_eps: float | None = None
    mode_spring: str | None = None
    k_anchor_spring: float | None = None
    k_anchor_occlusion: float | None = None
    anchor_spring_kind: str | None = None
    anchor_spring_alpha: float | None = None
    anchor_r0_points: float | None = None
    anchor_r0_lines: float | None = None
    eps_norm: float | None = None

    model_config = ConfigDict(extra="forbid")


class ForcesPublic(RootModel[Dict[str, Dict[str, TermParams]]]):
    root: Dict[str, Dict[str, TermParams]] = Field(default_factory=dict)


class ComputePublic(BaseModel):
    forces: ForcesPublic = Field(default_factory=ForcesPublic)

    model_config = ConfigDict(extra="forbid")


class SolverPublic(BaseModel):
    mode: str
    use_warmup: bool
    log_level: Literal["none", "info", "debug"] = "none"

    model_config = ConfigDict(extra="forbid")


class SolverTuningLBFGSB(BaseModel):
    lbfgs_pgtol: float = Field(gt=0)
    lbfgs_maxiter: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid")


class SolverTuningStop(BaseModel):
    ftol: float = Field(gt=0)
    xtol: float = Field(gt=0)
    max_stall_iters: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")


class SolverTuningSemiNewton(BaseModel):
    sn_max_outer: int = Field(ge=1)
    sn_dt: float = Field(gt=0)
    sn_hvp_eps: float = Field(gt=0)
    sn_cg_maxiter: int = Field(ge=1)
    sn_cg_rtol: float = Field(gt=0)
    sn_cg_atol: float = Field(ge=0)
    sn_lm0: float = Field(gt=0)
    sn_gtol: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class SolverTuningWarmup(BaseModel):
    steps: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid")


class SolverTuningHybrid(BaseModel):
    sn_max_outer: int = Field(ge=0)
    lbfgs_maxiter: int = Field(ge=1)
    sn_post_max_outer: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid")


class SolverTuning(BaseModel):
    lbfgsb: SolverTuningLBFGSB
    semi_newton: SolverTuningSemiNewton
    stop: SolverTuningStop
    warmup: SolverTuningWarmup
    hybrid: SolverTuningHybrid | None = None

    model_config = ConfigDict(extra="forbid")


class SolverInternalsStability(BaseModel):
    eps_norm: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class SolverInternals(BaseModel):
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
    solver: Solver
    public: ComputePublic = Field(default_factory=ComputePublic)

    model_config = ConfigDict(extra="forbid")


__all__ = ["TermParams", "ForcesPublic", "ComputePublic", "Compute"]


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
