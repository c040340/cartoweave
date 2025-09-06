"""Pydantic models for compute configuration."""
from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field


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
