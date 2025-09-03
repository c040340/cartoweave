"""High level interactive visualisation.

The original project uses a rather feature rich viewer implemented in a large
script.  For the purposes of the exercises in this repository we only require
something lightweight that exposes the essential ideas – a timeline of label
positions, per‑label force breakdown and an optional scalar field visualised as
either a heatmap or a 3‑D surface.  This module intentionally keeps the code
compact and easy to follow.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from .panels import (
    draw_field_panel,
    draw_force_panel,
    draw_info_panel,
    draw_layout,
)


def _as_vec2(a: Any) -> Optional[np.ndarray]:
    """Return ``a`` as ``(N, 2)`` float array or ``None``."""

    if a is None:
        return None
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    return None


def _compute_anchors(
    labels: Sequence[Dict[str, Any]],
    *,
    points: Any = None,
    lines: Any = None,
    areas: Any = None,
    sources: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    """Find anchor positions for ``labels``.

    ``sources`` may already contain anchor coordinates under the key
    ``"anchors_xy"`` or ``"anchor_xy"``.  When not provided, the anchors are
    inferred from the background geometry based on ``anchor_kind`` and
    ``anchor_index`` stored inside each label.  Only a few common formats are
    supported and any missing information results in ``NaN`` anchors.
    """

    if isinstance(sources, dict):
        arr = _as_vec2(sources.get("anchors_xy") or sources.get("anchor_xy"))
        if arr is not None:
            return arr

    pts = _as_vec2(points)
    anchors = []
    for lab in labels:
        kind = str(lab.get("anchor_kind", "")).lower()
        idx = lab.get("anchor_index")

        if kind == "point" and pts is not None and isinstance(idx, int) and 0 <= idx < len(pts):
            anchors.append(pts[idx])
        elif kind == "line" and isinstance(lines, (list, tuple)) and isinstance(idx, int) and 0 <= idx < len(lines):
            arr = _as_vec2(lines[idx])
            anchors.append(arr.mean(axis=0) if arr is not None else (np.nan, np.nan))
        elif kind == "area" and isinstance(areas, (list, tuple)) and isinstance(idx, int) and 0 <= idx < len(areas):
            arr = _as_vec2(areas[idx])
            anchors.append(arr.mean(axis=0) if arr is not None else (np.nan, np.nan))
        else:
            anchors.append((np.nan, np.nan))

    return np.asarray(anchors)


def interactive_view(
    traj: np.ndarray,
    labels: Sequence[Dict[str, Any]],
    rect_wh: np.ndarray,
    points: Any,
    lines: Any,
    areas: Any,
    W: float,
    H: float,
    *,
    force_getter: Optional[Callable[[int], Dict[str, np.ndarray]]] = None,
    source_getter: Optional[Callable[[int], Dict[str, Any]]] = None,
    metrics_getter: Optional[Callable[[int], Dict[str, Any]]] = None,
    field_getter: Optional[Callable[[int], Any]] = None,
    field_kind: str = "heatmap",
    field_cmap: str = "viridis",
) -> None:
    """Display an interactive layout viewer.

    Parameters are intentionally kept simple.  ``traj`` is expected to be an
    array with shape ``(T, N, 2)`` containing the centre positions of ``N``
    labels over ``T`` iterations.  ``rect_wh`` stores the widths and heights of
    the labels.  ``force_getter``/``source_getter``/``metrics_getter`` and
    ``field_getter`` are callables returning data for a given iteration.  They
    may be omitted, in which case the corresponding panels simply show
    placeholder information. ``field_kind`` selects between a 2‑D heatmap and a
    3‑D surface for scalar fields while ``field_cmap`` controls the colour map
    used for either representation.
    """

    traj = np.asarray(traj, dtype=float)
    rect_wh = np.asarray(rect_wh, dtype=float)
    T = len(traj)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax_layout = fig.add_subplot(gs[0, 0])
    ax_force = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 0])
    if field_kind == "3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax_field = fig.add_subplot(gs[1, 1], projection="3d")
    else:
        ax_field = fig.add_subplot(gs[1, 1])

    slider_ax = fig.add_axes([0.25, 0.02, 0.5, 0.04])
    slider = Slider(slider_ax, "iter", 0, T - 1, valinit=0, valstep=1)

    selected = 0
    patches = []

    def _get_forces(step: int) -> Dict[str, np.ndarray]:
        if callable(force_getter):
            out = force_getter(step)
            return out if isinstance(out, dict) else {}
        return {}

    def _get_sources(step: int) -> Dict[str, Any]:
        if callable(source_getter):
            out = source_getter(step)
            return out if isinstance(out, dict) else {}
        return {}

    def _get_metrics(step: int) -> Optional[Dict[str, Any]]:
        if callable(metrics_getter):
            out = metrics_getter(step)
            return out if isinstance(out, dict) else None
        return None

    def _get_field(step: int) -> Any:
        if callable(field_getter):
            return field_getter(step)
        return None

    def _update(step: int) -> None:
        nonlocal patches
        src = _get_sources(step)
        pts = src.get("points", points)
        lns = src.get("lines", lines)
        ars = src.get("areas", areas)
        anchors = _compute_anchors(labels, points=pts, lines=lns, areas=ars, sources=src)

        patches = draw_layout(
            ax_layout,
            traj[step],
            labels,
            rect_wh,
            frame_w=W,
            frame_h=H,
            points=pts,
            lines=lns,
            areas=ars,
            anchors=anchors,
        )

        forces = _get_forces(step)
        total = draw_force_panel(ax_force, forces, selected)

        metrics = _get_metrics(step)
        draw_info_panel(ax_info, forces, selected, total, metrics=metrics)

        field = _get_field(step)
        draw_field_panel(ax_field, field, field_kind, field_cmap)

        fig.canvas.draw_idle()

    def _on_pick(event):
        nonlocal selected
        for i, patch in patches:
            if event.artist is patch:
                selected = i
                _update(int(slider.val))
                break

    def _on_slider(val):
        _update(int(val))

    fig.canvas.mpl_connect("pick_event", _on_pick)
    slider.on_changed(_on_slider)

    _update(0)
    plt.show()


__all__ = ["interactive_view"]

