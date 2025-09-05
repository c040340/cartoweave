"""Dataclasses for scene generation configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteGenCfg:
    """Configuration for polyline route generation.

    All fields are expressed as scales relative to the frame diagonal ``D``.
    """
    min_vertex_spacing_scale: float = 0.010  # × D
    inset_margin_scale: float = 0.020  # × D
    segment_len_scale: float = 0.060  # × D
    angle_noise_sigma: float = 0.35  # radians


@dataclass(frozen=True)
class AreaGenCfg:
    """Configuration for polygon generation.

    ``inset_margin_scale`` and ``min_edge_spacing_scale`` are relative to the
    frame diagonal ``D``.  ``n_vertices_min`` and ``n_vertices_max`` bound the
    number of vertices sampled.
    """
    inset_margin_scale: float = 0.020  # × D
    min_edge_spacing_scale: float = 0.008  # × D
    n_vertices_min: int = 6
    n_vertices_max: int = 14

