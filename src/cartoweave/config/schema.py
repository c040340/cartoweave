"""Pydantic models describing the configuration schema."""
from __future__ import annotations

from typing import Dict, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError


class SolverPublic(BaseModel):
    mode: Literal["lbfgsb", "semi-newton", "simple", "hybrid"] = "lbfgsb"
    use_autocal: bool = False
    use_warmup: bool = False
    use_retry: bool = False
    use_topk_sources: bool = False
    profile: Literal["basic", "fast", "quality"] = "basic"
    force_level: str | int = Field("medium")

    model_config = ConfigDict(extra="forbid")


class MergeConfig(BaseModel):
    mode: Literal["sum", "softmax", "logsumexp", "weighted"] = "sum"
    temperature: float = 1.0


class NormalizeConfig(BaseModel):
    kind: Literal["l1", "l2", "max", "topk_norm"] = "l2"


class ThresholdConfig(BaseModel):
    abs: float = 1e-3
    rel: float = Field(0.0, ge=0.0, le=1.0)


class TopKConfig(BaseModel):
    enable: bool = False
    k: PositiveInt = Field(8, ge=1)
    min_share: float = Field(0.0, ge=0.0, le=1.0)


class LbfgsbConfig(BaseModel):
    m: PositiveInt = Field(10, gt=0)
    maxiter: PositiveInt = Field(100, gt=0)


class WarmupConfig(BaseModel):
    steps: int = 0


class RetryConfig(BaseModel):
    enable: bool = False


class AntiJumpConfig(BaseModel):
    max_step_px: float = 30.0


class ClampConfig(BaseModel):
    optimize_force_max: float | None = 1e6


class SolverTuning(BaseModel):
    merge: MergeConfig = MergeConfig()
    normalize: NormalizeConfig = NormalizeConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    topk: TopKConfig = TopKConfig()
    lbfgsb: LbfgsbConfig = LbfgsbConfig()
    clamp: ClampConfig = ClampConfig()
    warmup: WarmupConfig = WarmupConfig()
    retry: RetryConfig = RetryConfig()
    anti_jump: AntiJumpConfig = AntiJumpConfig()

    model_config = ConfigDict(extra="allow")


class EpsConfig(BaseModel):
    div: float = 1e-9
    sqrt: float = 1e-9
    proj: float = 1e-9


class ClipConfig(BaseModel):
    force_abs_max: float = 1e6
    energy_abs_max: float = 1e6


class TemperatureConfig(BaseModel):
    min: float = 1e-3
    max: float = 10.0


class SolverInternals(BaseModel):
    eps: EpsConfig = EpsConfig()
    clip: ClipConfig = ClipConfig()
    temperature: TemperatureConfig = TemperatureConfig()
    finite_check: Literal["warn", "error", "ignore"] = "warn"

    model_config = ConfigDict(extra="forbid")


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
    panels: VizPanels = VizPanels()
    colors: Dict[str, Any] = {}
    force_view: VizForceView = VizForceView()
    picker: VizPicker = VizPicker()
    surface3d: VizSurface3D = VizSurface3D()
    export: VizExport = VizExport()

    model_config = ConfigDict(extra="allow")


class SolverConfig(BaseModel):
    public: SolverPublic = SolverPublic()
    tuning: SolverTuning = SolverTuning()
    internals: SolverInternals = SolverInternals()

    model_config = ConfigDict(extra="allow")


class RootConfig(BaseModel):
    solver: SolverConfig = SolverConfig()
    viz: VizConfig = VizConfig()

    model_config = ConfigDict(extra="allow")


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate *cfg* against the configuration schema."""
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
    "SolverPublic",
    "SolverTuning",
    "SolverInternals",
    "VizConfig",
]

