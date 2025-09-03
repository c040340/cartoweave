"""Configuration defaults for visualisation parameters.

This module centralises all tunable styling options used by the helper
functions in :mod:`cartoweave.viz`. The settings are intentionally organised in
plain dictionaries so that users may easily override them if desired.
"""

from __future__ import annotations

from typing import Any, Dict

# NOTE: configuration is stored in a nested dictionary. Consumers import
# ``viz_config`` and read or modify values as needed.
viz_config: Dict[str, Any] = {
    "layout": {
        "colors": {
            "point": "rgba(31, 119, 180, 1)",  # blue
            "line": "rgba(255, 127, 14, 1)",  # orange
            "area": "rgba(44, 160, 44, 1)",  # green
            "label_fill": "rgba(242, 242, 255, 0.8)",  # light blue fill
            "label_edge": "rgba(43, 108, 176, 1)",  # blue outline
            "anchor_line": "rgba(136, 136, 136, 1)",  # grey line
            "anchor_marker_face": "rgba(26, 128, 230, 0.25)",  # translucent blue
            "anchor_marker_edge": "rgba(43, 108, 176, 1)",  # blue edge
        },
        "line_width": 1.0,
        "area_edge_width": 1.0,
        "area_face_alpha": 0.15,
        "anchor_marker_size": 5.0,
        "label_edge_width": 1.0,
        "label_fontsize": 9,
    },
    "forces": {
        "colors": {
            "focus": "rgba(31, 119, 180, 1)",  # blue
            "boundary": "rgba(255, 127, 14, 1)",  # orange
            "label_label": "rgba(44, 160, 44, 1)",  # green
            "anchor": "rgba(214, 39, 40, 1)",  # red
            "total": "rgba(0, 0, 0, 1)",  # black
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
