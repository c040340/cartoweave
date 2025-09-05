"""Normalized view for legacy solver attributes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


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
def make_layout_config(core: Dict[str, Any]) -> LayoutConfig:
    """Build ``LayoutConfig`` from a raw configuration dictionary.

    This helper mirrors a subset of the legacy dataclass view.  Missing keys are
    filled with conservative defaults.
    """
    solver = core.get("solver", {})
    t = solver.get("tuning", {})
    ld = core.get("logging_debug", {})
    stopping = t.get("stopping", {})
    lbfgsb = t.get("lbfgsb", {})
    warmup = t.get("warmup", {})
    retry = t.get("retry", {})
    return LayoutConfig(
        gtol_ref_kind=stopping.get("gtol_ref_kind", "p75"),
        gtol_abs=stopping.get("gtol_abs", 1e-2),
        gtol_rel=stopping.get("gtol_rel", 2e-3),
        gtol_cap=stopping.get("gtol_cap"),
        lbfgsb_restarts=lbfgsb.get("restarts", 0),
        factr=lbfgsb.get("factr", 1.0),
        lbfgs_m=lbfgsb.get("m", 0),
        maxiter=lbfgsb.get("maxiter", 0),
        maxfun=lbfgsb.get("maxfun", 0),
        maxls=lbfgsb.get("maxls", 0),
        lbfgs_disp=lbfgsb.get("disp", False),
        warmup_steps=warmup.get("steps", 0),
        warmup_pixel_equiv=warmup.get("pixel_equiv", 0.0),
        warmup_step_cap=warmup.get("step_cap", 0.0),
        restart_noise_px=retry.get("noise_px", 0.0),
        restart_noise_ratio=retry.get("noise_ratio", 0.0),
        restart_noise_seed=retry.get("noise_seed", 0),
        debug_solver=ld.get("debug_solver", False),
        debug_check=ld.get("debug_check", False),
    )

__all__ = ["make_layout_config", "LayoutConfig"]
