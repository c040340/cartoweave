import numpy as np
from cartoweave.layout_utils.geometry import (
    poly_centroid,
    polyline_uniform_arc_centroid,
    project_point_to_polyline,
    area_anchor_from_centroid_nearest_edge,
)

__all__ = ["anchor_xy"]


def line_midpoint_xy(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, float)
    if pts.shape[0] == 0:
        return np.zeros(2, float)
    return 0.5 * (pts[0] + pts[-1])


def line_endpoints_mid_xy(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, float)
    C = polyline_uniform_arc_centroid(pts, step_len=4.0)
    Q, _, _ = project_point_to_polyline(C, pts)
    return Q


def polygon_centroid_xy(poly: np.ndarray) -> np.ndarray:
    cx, cy = poly_centroid(np.asarray(poly, float))
    return np.array([cx, cy], float)


def polygon_bbox_center_xy(poly: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly, float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return 0.5 * (mins + maxs)


def nearest_edge_point_to_centroid(poly: np.ndarray) -> np.ndarray:
    info = area_anchor_from_centroid_nearest_edge(np.asarray(poly, float))
    return np.array([info["qx"], info["qy"]], float)


def anchor_xy(scene, anchor, line_anchor_mode: str = "midpoint", area_anchor_mode: str = "centroid"):
    """Map ``AnchorSpec(kind,index)`` to ``np.array([x,y])`` using existing routines."""

    # allow legacy call order ``anchor_xy(anchor, scene)``
    if not hasattr(scene, "points") and (
        isinstance(scene, dict) and ("kind" in scene or "anchor_kind" in scene)
    ):
        scene, anchor = anchor, scene

    if anchor is None:
        return None
    if isinstance(anchor, dict):
        k = anchor.get("kind") or anchor.get("anchor_kind")
        idx = anchor.get("index", anchor.get("anchor_index", -1))
    else:
        k = getattr(anchor, "kind", None)
        idx = getattr(anchor, "index", -1)
    idx = int(idx if idx is not None else -1)

    if k == "point":
        pts = np.asarray(getattr(scene, "points", []), float)
        if 0 <= idx < len(pts):
            return np.asarray(pts[idx], float)
        return None

    if k == "line":
        lines = getattr(scene, "lines", [])
        if 0 <= idx < len(lines):
            coords = np.asarray(lines[idx], float)
            if line_anchor_mode == "endproj":
                return line_endpoints_mid_xy(coords)
            return line_midpoint_xy(coords)
        return None

    if k == "area":
        areas = getattr(scene, "areas", [])
        if 0 <= idx < len(areas):
            ar = areas[idx]
            poly = ar.get("exterior") if isinstance(ar, dict) else ar
            poly = np.asarray(poly, float)
            if area_anchor_mode == "center":
                return polygon_bbox_center_xy(poly)
            if area_anchor_mode == "nearest_edge":
                return nearest_edge_point_to_centroid(poly)
            return polygon_centroid_xy(poly)
        return None

    return None
