"""Helper utilities for reading physics term toggles and focus parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

@dataclass(frozen=True)
class FocusParams:
    shape: str
    center_px: Tuple[float, float]
    axes_px: Tuple[float, float]
    angle_deg: float
    falloff: Dict[str, float]


def get_enabled_terms(cfg: Dict[str, Any]) -> List[str]:
    """Return list of enabled physics terms."""
    enabled = cfg.get("physics", {}).get("tuning", {}).get("enabled", {})
    return [name for name, flag in enabled.items() if flag]


def get_focus_params(cfg: Dict[str, Any], frame_size: Tuple[int, int]) -> FocusParams:
    """Map focus parameters to pixel space."""
    tuning = cfg.get("physics", {}).get("tuning", {})
    f = tuning.get("focus", {})
    w, h = frame_size
    cn = f.get("center_norm", [0.5, 0.5])
    cx = cn[0] * w
    cy = cn[1] * h
    return FocusParams(
        shape=f.get("shape", "ellipse"),
        center_px=(cx, cy),
        axes_px=tuple(f.get("axes_px", [w, h])),
        angle_deg=float(f.get("angle_deg", 0.0)),
        falloff=dict(f.get("falloff", {})),
    )

__all__ = ["get_enabled_terms", "get_focus_params", "FocusParams"]
