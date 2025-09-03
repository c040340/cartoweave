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
            "point": "#1F77B4FF",  # blue
            "line": "#FF7F0EFF",  # orange
            "area": "#2CA02CFF",  # green
            "label_fill": "#F2F2FFCC",  # light blue fill
            "label_edge": "#2B6CB0FF",  # blue outline
            "anchor_line": "#888888FF",  # grey line
            "anchor_marker_face": "#1A80E640",  # translucent blue
            "anchor_marker_edge": "#2B6CB0FF",  # blue edge
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
            "focus": "#1F77B4FF",  # blue
            "boundary": "#FF7F0EFF",  # orange
            "label_label": "#2CA02CFF",  # green
            "anchor": "#D62728FF",  # red
            "total": "#777777FF",  # black
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
