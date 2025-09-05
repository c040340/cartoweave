"""Pydantic models describing the configuration schema."""
from __future__ import annotations

from typing import Dict, Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    model_validator,
)


class SolverPublic(BaseModel):
    mode: Literal["lbfgsb", "simple", "hybrid", "newton"] = "lbfgsb"
    use_autocal: bool = False
    use_warmup: bool = False
    use_retry: bool = False
    use_topk_sources: bool = False
    profile: Literal[
        "basic",
        "fast",
        "quality",
        "balanced",
        "stable",
        "cinematic",
    ] = "basic"
    force_level: str | int = Field("medium")

    model_config = ConfigDict(extra="forbid")


class MergeConfig(BaseModel):
    mode: Literal["sum", "softmax", "logsumexp", "weighted"] = "sum"
    temperature: float = 1.0

    model_config = ConfigDict(extra="forbid")


class NormalizeConfig(BaseModel):
    kind: Literal["l1", "l2", "max", "topk_norm"] = "l2"

    model_config = ConfigDict(extra="forbid")


class ThresholdConfig(BaseModel):
    abs: float | None = Field(None, ge=0.0)
    rel: float | None = Field(None, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class TopKConfig(BaseModel):
    enable: bool = False
    k: PositiveInt = Field(8, ge=1)
    min_share: float = Field(0.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class LbfgsbConfig(BaseModel):
    m: PositiveInt = Field(10, gt=0)
    maxiter: PositiveInt = Field(100, ge=1)

    model_config = ConfigDict(extra="forbid")


class StoppingConfig(BaseModel):
    gtol: float = Field(1e-5, gt=0.0)

    model_config = ConfigDict(extra="forbid")


class WarmupConfig(BaseModel):
    steps: int = Field(0, ge=0)

    model_config = ConfigDict(extra="forbid")


class RetryConfig(BaseModel):
    enable: bool = False

    model_config = ConfigDict(extra="forbid")


class AntiJumpConfig(BaseModel):
    step_cap_px: float | None = Field(30.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class ClampConfig(BaseModel):
    optimize_force_max: float | None = Field(1e6, gt=0.0)

    model_config = ConfigDict(extra="forbid")


class AnchorConfig(BaseModel):
    k_spring: float = Field(5.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class TermConfig(BaseModel):
    weight: float = Field(1.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class SolverTuning(BaseModel):
    merge: MergeConfig = Field(default_factory=MergeConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    topk: TopKConfig = Field(default_factory=TopKConfig)
    lbfgsb: LbfgsbConfig = Field(default_factory=LbfgsbConfig)
    stopping: StoppingConfig = Field(default_factory=StoppingConfig)
    clamp: ClampConfig = Field(default_factory=ClampConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    anti_jump: AntiJumpConfig = Field(default_factory=AntiJumpConfig)
    anchor: AnchorConfig = Field(default_factory=AnchorConfig)
    terms: Dict[str, TermConfig] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class EpsConfig(BaseModel):
    div: float = Field(1e-9, gt=0.0)
    sqrt: float = Field(1e-9, gt=0.0)
    proj: float = Field(1e-9, gt=0.0)

    model_config = ConfigDict(extra="forbid")


class ClipConfig(BaseModel):
    force_abs_max: float = Field(1e6, gt=0.0)
    energy_abs_max: float = Field(1e6, gt=0.0)

    model_config = ConfigDict(extra="forbid")


class TemperatureConfig(BaseModel):
    min: float = Field(1e-3, gt=0.0)
    max: float = Field(10.0, gt=0.0)

    @model_validator(mode="after")
    def _check_bounds(cls, data: "TemperatureConfig") -> "TemperatureConfig":
        if data.min > data.max:
            raise ValueError("min must be <= max")
        return data

    model_config = ConfigDict(extra="forbid")


class SolverInternals(BaseModel):
    eps: EpsConfig = Field(default_factory=EpsConfig)
    clip: ClipConfig = Field(default_factory=ClipConfig)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)
    finite_check: Literal["warn", "raise", "ignore"] = "warn"
    stability: "StabilityConfig" = Field(default_factory=lambda: StabilityConfig())

    model_config = ConfigDict(extra="forbid")


class StabilityConfig(BaseModel):
    softplus_limit: float = 40.0
    logsumexp_floor: float = -40.0
    eps_sigma: float = 1.0e-3
    eps_norm: float = 1.0e-12
    exp_clip: float = 40.0
    area_eps: float = 1.0e-12

    model_config = ConfigDict(extra="forbid")


class AnchorSpringCfg(BaseModel):
    k: float = 6.0
    r0: float = 12.0
    damping: float = 0.0
    max_norm_cap: float | None = None


class FocusCfg(BaseModel):
    k: float = 0.5
    sigma: float = 40.0


class BoundaryCfg(BaseModel):
    k: float = 2.0
    softness: float = 4.0


class RepulseCfg(BaseModel):
    k: float = 0.3
    d0: float = 10.0
    exponent: float = 2.0


class InsideCfg(BaseModel):
    k: float = 0.2
    margin: float = 4.0


class AreaCrossCfg(BaseModel):
    k: float = 1.2
    sigma: float = 6.0


class AreaEmbedCfg(BaseModel):
    k: float = 0.8
    sigma: float = 6.0
    edge_bias: float = 0.0


class TermsCfg(BaseModel):
    anchor: dict = Field(default_factory=lambda: {"spring": AnchorSpringCfg()})
    focus: FocusCfg = FocusCfg()
    boundary: BoundaryCfg = BoundaryCfg()
    label_label_repulse: RepulseCfg = RepulseCfg()
    line_label_repulse: RepulseCfg = RepulseCfg()
    point_label_repulse: RepulseCfg = RepulseCfg()
    label_label_inside: InsideCfg = InsideCfg()
    line_label_inside: InsideCfg = InsideCfg()
    point_label_inside: InsideCfg = InsideCfg()
    area_cross: AreaCrossCfg = AreaCrossCfg()
    area_embed: AreaEmbedCfg = AreaEmbedCfg()

    model_config = ConfigDict(extra="forbid")


class RouteGenCfg(BaseModel):
    mean_length_scale: float = 0.25
    k_sigma_bound: int = 5
    max_retry: int = 50
    min_vertex_spacing_scale: float = 0.01
    min_edge_margin_scale: float = 0.02
    lower_bound_scale: float = 0.02
    upper_bound_scale: float = 0.60


class AreaGenCfg(BaseModel):
    mean_area_scale: float = 0.05
    k_sigma_bound: int = 5
    max_retry: int = 50
    min_vertex_spacing_scale: float = 0.01
    min_edge_margin_scale: float = 0.02
    lower_bound_scale: float = 0.01
    upper_bound_scale: float = 0.50


class DataRandomCounts(BaseModel):
    n_points: int = 8
    n_lines: int = 3
    n_areas: int = 2


class DataRandomFrame(BaseModel):
    width: int = 1920
    height: int = 1080


class DataRandomCfg(BaseModel):
    frame: DataRandomFrame = DataRandomFrame()
    counts: DataRandomCounts = DataRandomCounts()
    route_gen: RouteGenCfg = RouteGenCfg()
    area_gen: AreaGenCfg = AreaGenCfg()


class DataConfig(BaseModel):
    random: DataRandomCfg = Field(default_factory=DataRandomCfg)


class VizPanels(BaseModel):
    layout: bool = True
    forces: bool = False

    model_config = ConfigDict(extra="allow")


class VizForceView(BaseModel):
    show_vectors: bool = True
    clamp_force_max: float | None = 1000.0

    model_config = ConfigDict(extra="allow")


class VizPicker(BaseModel):
    enable: bool = False

    model_config = ConfigDict(extra="allow")


class VizSurface3D(BaseModel):
    enable: bool = False

    model_config = ConfigDict(extra="allow")


class VizExport(BaseModel):
    enable: bool = False

    model_config = ConfigDict(extra="allow")


class VizConfig(BaseModel):
    panels: VizPanels = Field(default_factory=VizPanels)
    colors: Dict[str, Any] = Field(default_factory=dict)
    force_view: VizForceView = Field(default_factory=VizForceView)
    picker: VizPicker = Field(default_factory=VizPicker)
    surface3d: VizSurface3D = Field(default_factory=VizSurface3D)
    export: VizExport = Field(default_factory=VizExport)

    model_config = ConfigDict(extra="allow")


class SolverConfig(BaseModel):
    public: SolverPublic = Field(default_factory=SolverPublic)
    tuning: SolverTuning = Field(default_factory=SolverTuning)
    internals: SolverInternals = Field(default_factory=SolverInternals)

    model_config = ConfigDict(extra="forbid")


class RootConfig(BaseModel):
    solver: Dict[str, Any] = Field(default_factory=dict)
    viz: VizConfig = Field(default_factory=VizConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    terms: TermsCfg = Field(default_factory=TermsCfg)

    model_config = ConfigDict(extra="forbid")


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate *cfg* against the configuration schema."""
    def _scan(data: Mapping[str, Any], path: str = "") -> None:
        for key, value in data.items():
            full_key = f"{path}.{key}" if path else key
            if "." in key and not full_key.startswith("viz."):
                raise ValueError(
                    f"Invalid configuration: legacy dotted key '{full_key}'"
                )
            if full_key.endswith("term_weights"):
                raise ValueError(
                    "Invalid configuration: 'term_weights' is no longer supported"
                )
            if isinstance(value, Mapping):
                _scan(value, full_key)

    _scan(cfg)

    topk = cfg.get("solver", {}).get("topk", {})
    ms = topk.get("min_share")
    if ms is not None and ms > 1.0:
        raise ValueError("Invalid configuration: solver.topk.min_share > 1")
    try:
        RootConfig(**cfg)
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        messages = []
        for err in exc.errors():
            path = ".".join(str(p) for p in err["loc"])
            messages.append(f"{path}: {err['msg']}")
        raise ValueError("Invalid configuration: " + "; ".join(messages))


__all__ = [
    "validate_config",
    "VizConfig",
    "DataRandomCfg",
    "TermsCfg",
]

