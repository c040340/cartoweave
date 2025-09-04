"""Visualisation helpers for CartoWeave.

The :func:`interactive_view` function provides a lightweight interactive
viewer that mirrors the behaviour of the much larger original script while
remaining easy to understand.
"""

from typing import Any, Dict

from .view import interactive_view

# Minimal stub matching the previous public API.  Rich configuration is handled
# via external files in the new loader; the viewer defaults to empty settings
# when used directly.
viz_config: Dict[str, Any] = {
    "info": {"title_fontsize": 10, "row_main_fontsize": 8, "row_component_fontsize": 8},
    "forces": {
        "colors": {
            "ll.rect": "#1f77b4",
            "boundary.wall": "#ff7f0e",
            "anchor.spring": "#2ca02c",
        },
        "component_arrow_scale": 10.0,
        "component_arrow_lw": 1.0,
        "component_fontsize": 8,
        "total_arrow_scale": 12.0,
        "total_arrow_lw": 1.5,
    },
    "layout": {
        "colors": {
            "point": "#000000",
            "line": "#000000",
            "area": "#000000",
            "label_fill": "#FFFFFF",
            "label_edge": "#000000",
            "anchor_line": "#000000",
            "anchor_marker_face": "#FFFFFF",
            "anchor_marker_edge": "#000000",
        },
        "line_width": 1.0,
        "area_face_alpha": 0.3,
        "area_edge_width": 1.0,
        "label_edge_width": 1.0,
        "label_fontsize": 10,
        "anchor_marker_size": 4.0,
    },
}

__all__ = ["interactive_view", "viz_config"]

