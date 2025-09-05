from copy import deepcopy
from typing import Any, Dict, Mapping

from cartoweave.utils.dict_merge import deep_update

# Minimal viewer defaults used when no YAML configuration is supplied.
VIZ_DEFAULTS: Dict[str, Any] = {
    "panels": {"layout": True, "forces": True, "info": True, "field": True},
    "field": {"mode": "heatmap", "resolution": 128, "cmap": "viridis"},
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
            "points": "#000000",
            "lines": "#000000",
            "areas": "#000000",
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


def merge_defaults(user_viz: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return ``VIZ_DEFAULTS`` merged with ``user_viz``."""
    base = deepcopy(VIZ_DEFAULTS)
    if not user_viz:
        return base
    return deep_update(base, user_viz)
