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
        # ``dict.get`` with ``or`` cannot be used here because NumPy arrays do
        # not define a truth value.  Explicitly check both keys instead.
        arr = sources.get("anchors_xy")
        if arr is None:
            arr = sources.get("anchor_xy")
        arr = _as_vec2(arr)
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


def _label_name(lab: Dict[str, Any], index: int) -> str:
    """Return a compact identifier for *lab*.

    Falls back to the numeric index when no explicit identifier is stored in
    the label dictionary.
    """

    for key in ("uid", "id", "name"):
        val = lab.get(key)
        if isinstance(val, str) and val:
            return val
    return str(index)


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
    boundaries: Optional[Sequence[int]] = None,
    actions: Optional[Sequence[Any]] = None,
) -> None:
    """Display an interactive layout viewer.

    Parameters are intentionally kept simple. ``traj`` is expected to be an
    array with shape ``(T, N, 2)`` containing the centre positions of ``N``
    labels over ``T`` iterations. ``rect_wh`` stores the widths and heights of
    the labels. ``force_getter``/``source_getter``/``metrics_getter`` and
    ``field_getter`` are callables returning data for a given iteration. They
    may be omitted, in which case the corresponding panels simply show
    placeholder information. ``field_kind`` selects between a 2‑D heatmap and a
    3‑D surface for scalar fields while ``field_cmap`` controls the colour map
    used for either representation.

    ``boundaries`` and ``actions`` describe high level actions in the timeline.
    When action information is available (either explicit ``actions`` or
    ``boundaries`` describing iteration ranges) an additional slider lets the
    user switch between actions while constraining the iteration slider to the
    corresponding range.
    """

    traj = np.asarray(traj, dtype=float)
    rect_wh = np.asarray(rect_wh, dtype=float)
    T = len(traj)

    # ------------------------------------------------------------------
    # Layout of the figure
    # ------------------------------------------------------------------
    # The overall view is split into a tall main area and a slim bar at the
    # bottom that hosts the iteration slider and timeline actions.
    fig = plt.figure(figsize=(22, 10))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[28, 1],
        left=0.05, right=0.97, top=0.95, bottom=0.05
    )

    # --- main area -----------------------------------------------------
    main = outer[0].subgridspec(1, 3, width_ratios=[5.5, 5, 7.5])
    ax_layout = fig.add_subplot(main[0, 0])

    centre = main[0, 1].subgridspec(2, 1, height_ratios=[2.5, 2])
    ax_force = fig.add_subplot(centre[0, 0])
    ax_info = fig.add_subplot(centre[1, 0])


    if field_kind == "3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax_field = fig.add_subplot(main[0, 2], projection="3d")
    else:
        ax_field = fig.add_subplot(main[0, 2])

    # --- bottom bar ----------------------------------------------------
    bottom = outer[1].subgridspec(1, 2, width_ratios=[2, 3], wspace=0.15)
    slider_ax = fig.add_subplot(bottom[0, 0])
    slider_ax.set_xticks([])
    slider_ax.set_yticks([])
    slider = Slider(slider_ax, "iter", 0, T - 1, valinit=0, valstep=1)
    slider.valtext.set_text(f"1/{T}")

    action_ax = fig.add_subplot(bottom[0, 1])
    action_ax.set_xticks([])
    action_ax.set_yticks([])
    action_slider: Optional[Slider] = None
    n_actions = 0
    if actions is not None:
        try:
            n_actions = len(actions)
        except TypeError:
            actions = list(actions)
            n_actions = len(actions)
    if boundaries is not None:
        try:
            boundaries = list(boundaries)
        except TypeError:
            boundaries = list(boundaries)
        if n_actions == 0:
            n_actions = max(len(boundaries) - 1, 0)
    if n_actions > 0:
        action_slider = Slider(action_ax, "action", 0, n_actions - 1, valinit=0, valstep=1)
        action_slider.valtext.set_text("0")
    else:
        action_ax.set_visible(False)

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

    def _sum_all_forces(forces: Dict[str, np.ndarray]) -> np.ndarray:
        """Sum all force components over all labels and return the net vector."""
        total = None
        for arr in forces.values():
            a = _as_vec2(arr)
            if a is not None:
                total = a if total is None else total + a
        if total is None:
            return np.zeros(2, dtype=float)
        return np.sum(total, axis=0)

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
        label_id = _label_name(labels[selected], selected)
        label_total = draw_force_panel(ax_force, forces, selected, title=label_id)

        # --- global forces --------------------------------------------------
        vec_all = _sum_all_forces(forces)
        g_mag = float(np.hypot(vec_all[0], vec_all[1]))
        g_ang = float(np.degrees(np.arctan2(vec_all[1], vec_all[0])))

        d_pair = None
        if step > 0:
            prev_forces = _get_forces(step - 1)
            prev_vec = _sum_all_forces(prev_forces)
            prev_mag = float(np.hypot(prev_vec[0], prev_vec[1]))
            d_abs = prev_mag - g_mag
            d_rel = d_abs / prev_mag if prev_mag > 0 else 0.0
            d_pair = (d_abs, d_rel)

        metrics = _get_metrics(step)
        draw_info_panel(
            ax_info,
            forces,
            selected,
            label_total,
            (g_mag, g_ang),
            d_force=d_pair,
            metrics=metrics,
        )

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
        step = int(val)
        _update(step)
        slider.valtext.set_text(f"{step + 1}/{T}")

    def _on_action(val):
        if (
            boundaries is None
            or action_slider is None
            or len(boundaries) < 2
        ):
            return
        idx = int(val)
        lo = boundaries[idx]
        hi = boundaries[idx + 1] - 1
        slider.valmin = lo
        slider.valmax = hi
        slider.ax.set_xlim(lo, hi)
        slider.set_val(lo)

    fig.canvas.mpl_connect("pick_event", _on_pick)
    slider.on_changed(_on_slider)
    if action_slider is not None:
        action_slider.on_changed(_on_action)
        _on_action(0)

    _update(0)
    plt.show()


__all__ = ["interactive_view"]

