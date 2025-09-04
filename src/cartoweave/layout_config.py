"""Normalized view for legacy solver attributes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config_loader import CoreConfig


@dataclass(frozen=True)
class LayoutConfig:
    gtol_ref_kind: str
    gtol_abs: float
    gtol_rel: float
    gtol_cap: Any
    lbfgsb_restarts: int
    factr: float
    lbfgs_m: int
    maxiter: int
    maxfun: int
    maxls: int
    lbfgs_disp: bool
    warmup_steps: int
    warmup_pixel_equiv: float
    warmup_step_cap: float
    restart_noise_px: float
    restart_noise_ratio: float
    restart_noise_seed: int
    debug_solver: bool
    debug_check: bool


def make_layout_config(core: CoreConfig) -> LayoutConfig:
    t = core.solver.tuning
    ld = core.logging_debug
    return LayoutConfig(
        gtol_ref_kind=t.stopping.gtol_ref_kind,
        gtol_abs=t.stopping.gtol_abs,
        gtol_rel=t.stopping.gtol_rel,
        gtol_cap=t.stopping.gtol_cap,
        lbfgsb_restarts=t.lbfgsb.restarts,
        factr=t.lbfgsb.factr,
        lbfgs_m=t.lbfgsb.m,
        maxiter=t.lbfgsb.maxiter,
        maxfun=t.lbfgsb.maxfun,
        maxls=t.lbfgsb.maxls,
        lbfgs_disp=t.lbfgsb.disp,
        warmup_steps=t.warmup.steps,
        warmup_pixel_equiv=t.warmup.pixel_equiv,
        warmup_step_cap=t.warmup.step_cap,
        restart_noise_px=t.retry.noise_px,
        restart_noise_ratio=t.retry.noise_ratio,
        restart_noise_seed=t.retry.noise_seed,
        debug_solver=ld.debug_solver,
        debug_check=ld.debug_check,
    )

__all__ = ["make_layout_config", "LayoutConfig"]
