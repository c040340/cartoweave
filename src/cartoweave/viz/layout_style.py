from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LayoutStyle:
    """Style values for the layout panel.

    The fields map directly to existing YAML keys documented below:
      - layout.colors.points
      - layout.colors.lines
      - layout.colors.areas
      - layout.colors.label_fill
      - layout.colors.label_edge
      - layout.colors.anchor_line
      - layout.colors.anchor_marker_face
      - layout.colors.anchor_marker_edge
      - layout.line_width
      - layout.area_face_alpha
      - layout.area_edge_width
      - layout.label_edge_width
      - layout.label_fontsize
      - layout.anchor_marker_size
    """

    # basic axes options (no YAML keys exist yet)
    background: str = "#FFFFFFFF"
    grid_color: str = "#DDDDDDFF"
    show_grid: bool = True
    equal_aspect: bool = True
    invert_y: bool = True
    grid_lw: float = 0.5
    point_size: float = 18.0

    # values sourced from YAML
    points_color: str = "#000000"
    lines_color: str = "#000000"
    line_width: float = 1.0
    areas_face_color: str = "#000000"
    areas_face_alpha: float = 0.3
    areas_edge_color: str = "#000000"
    areas_edge_width: float = 1.0
    labels_fill_color: str = "#FFFFFF"
    labels_edge_color: str = "#000000"
    label_edge_width: float = 1.0
    label_fontsize: float = 10.0
    anchor_line_color: str = "#000000"
    anchor_marker_face: str = "#FFFFFF"
    anchor_marker_edge: str = "#000000"
    anchor_marker_size: float = 4.0


REQUIRED_KEYS = [
    "layout.colors.points",
    "layout.colors.lines",
    "layout.colors.areas",
    "layout.colors.label_fill",
    "layout.colors.label_edge",
    "layout.colors.anchor_line",
    "layout.colors.anchor_marker_face",
    "layout.colors.anchor_marker_edge",
    "layout.line_width",
    "layout.area_face_alpha",
    "layout.area_edge_width",
    "layout.label_edge_width",
    "layout.label_fontsize",
    "layout.anchor_marker_size",
]


def _get(d: Dict[str, Any], dotted: str):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required key: {dotted}")
        cur = cur[k]
    return cur


def get_layout_style_from_cfg(cfg: Dict[str, Any]) -> LayoutStyle:
    """Return :class:`LayoutStyle` populated from ``cfg``.

    Only reads existing keys listed in :data:`REQUIRED_KEYS`.  Missing keys raise
    :class:`KeyError` with a clear message to aid debugging.
    """

    missing = []
    for k in REQUIRED_KEYS:
        try:
            _get(cfg, k)
        except KeyError:
            missing.append(k)
    if missing:
        raise KeyError(
            "Missing required viz layout keys: " + ", ".join(missing)
        )

    return LayoutStyle(
        points_color=_get(cfg, "layout.colors.points"),
        lines_color=_get(cfg, "layout.colors.lines"),
        line_width=_get(cfg, "layout.line_width"),
        areas_face_color=_get(cfg, "layout.colors.areas"),
        areas_face_alpha=_get(cfg, "layout.area_face_alpha"),
        areas_edge_color=_get(cfg, "layout.colors.areas"),
        areas_edge_width=_get(cfg, "layout.area_edge_width"),
        labels_fill_color=_get(cfg, "layout.colors.label_fill"),
        labels_edge_color=_get(cfg, "layout.colors.label_edge"),
        label_edge_width=_get(cfg, "layout.label_edge_width"),
        label_fontsize=_get(cfg, "layout.label_fontsize"),
        anchor_line_color=_get(cfg, "layout.colors.anchor_line"),
        anchor_marker_face=_get(cfg, "layout.colors.anchor_marker_face"),
        anchor_marker_edge=_get(cfg, "layout.colors.anchor_marker_edge"),
        anchor_marker_size=_get(cfg, "layout.anchor_marker_size"),
    )
