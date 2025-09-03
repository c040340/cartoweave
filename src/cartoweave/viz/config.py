"""Configuration defaults for visualisation parameters.

This module centralises all tunable styling options used by the helper
functions in :mod:`cartoweave.viz`.  The settings are intentionally organised in
plain dictionaries so that users may easily override them if desired.
"""

from __future__ import annotations

from typing import Any, Dict

# NOTE: configuration is stored in a nested dictionary.  Consumers import
# ``viz_config`` and read or modify values as needed.

viz_config: Dict[str, Any] = {
    "layout": {
        "point_color": "#1f77b4",
        "line_color": "#ff7f0e",
        "line_width": 1.0,
        "area_color": "#2ca02c",
        "area_edge_width": 1.0,
        "area_face_alpha": 0.15,
        "label_fill_color": (0.95, 0.95, 1.0, 0.8),
        "label_edge_color": "#2B6CB0",
        "label_edge_width": 1.0,
        "anchor_line_color": "#888888",
        "anchor_marker_face": (0.1, 0.5, 0.9, 0.25),
        "anchor_marker_edge": "#2B6CB0",
        "anchor_marker_size": 5.0,
        "label_fontsize": 9,
    },
    "forces": {
        "colors": {
            "focus": "#1f77b4",
            "boundary": "#ff7f0e",
            "label_label": "#2ca02c",
            "anchor": "#d62728",
            "total": "#000000",
        },
        "component_arrow_lw": 2.0,
        "component_arrow_scale": 15,
        "component_fontsize": 6,
        "total_arrow_lw": 2.5,
        "total_arrow_scale": 18,
    },
    "info": {
        "title_fontsize": 10,
        "row_main_fontsize": 9,
        "row_component_fontsize": 8,
    },
}

__all__ = ["viz_config"]

