"""Helper utilities for reading physics term toggles and focus parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..config_loader import CoreConfig


@dataclass(frozen=True)
class FocusParams:
    shape: str
    center_px: Tuple[float, float]
    axes_px: Tuple[float, float]
    angle_deg: float
    falloff: Dict[str, float]


def get_enabled_terms(cfg: CoreConfig) -> List[str]:
    """Return list of enabled physics terms."""
    enabled = cfg.physics.tuning.enabled
    terms = [name for name, flag in enabled.__dict__.items() if flag]
    return terms


def get_focus_params(cfg: CoreConfig, frame_size: Tuple[int, int]) -> FocusParams:
    f = cfg.physics.tuning.focus
    w, h = frame_size
    cx = f.center_norm[0] * w
    cy = f.center_norm[1] * h
    return FocusParams(
        shape=f.shape,
        center_px=(cx, cy),
        axes_px=tuple(f.axes_px),
        angle_deg=f.angle_deg,
        falloff=dict(f.falloff.__dict__),
    )

__all__ = ["get_enabled_terms", "get_focus_params", "FocusParams"]
