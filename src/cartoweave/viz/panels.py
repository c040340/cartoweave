"""Utility routines for simple visualisation panels.

This module provides small helpers used by :mod:`cartoweave.viz.view`.
The functions deliberately implement only the features required by the
examples and prioritise clarity over completeness.  They accept plain data
structures (mostly ``numpy`` arrays and ``dict`` objects) and draw using
Matplotlib primitives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Colour map for force components.  The keys are expected to match those
# produced by ``force_getter`` functions.
FORCE_COLORS: Dict[str, str] = {
    "focus": "#1f77b4",
    "boundary": "#ff7f0e",
    "label_label": "#2ca02c",
    "anchor": "#d62728",
    "total": "#000000",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_vec2(a: Any) -> Optional[np.ndarray]:
    """Convert ``a`` to an ``(N, 2)`` float array if possible."""

    if a is None:
        return None
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    return None


def _label_text(lab: Dict[str, Any], index: int) -> str:
    """Return a short identifier for *lab* used in the layout view."""

    for key in ("uid", "id", "name"):
        val = lab.get(key)
        if isinstance(val, str) and val:
            return val
    return f"{index}"


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def draw_layout(
    ax: plt.Axes,
    pos: np.ndarray,
    labels: Sequence[Dict[str, Any]],
    rect_wh: np.ndarray,
    *,
    frame_w: float,
    frame_h: float,
    points: Any = None,
    lines: Any = None,
    areas: Any = None,
    anchors: Optional[np.ndarray] = None,
) -> List[Tuple[int, Rectangle]]:
    """Render the main layout panel.

    The function returns a list of ``(index, Rectangle)`` tuples.  The
    rectangles are marked as pickable so that callers can detect clicks and
    change the selected label.
    """

    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(0, frame_w)
    ax.set_ylim(frame_h, 0)  # origin in the top-left corner
    ax.set_facecolor("white")

    # add simple tick marks to help gauge scale
    ax.set_xticks(np.linspace(0, frame_w, 5))
    ax.set_yticks(np.linspace(0, frame_h, 5))
    ax.grid(True, color="#DDDDDD", lw=0.5)

    # --- background geometry -------------------------------------------------
    pts = _as_vec2(points)
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], c="#444444", s=18, zorder=1)

    if isinstance(lines, (list, tuple)):
        for pl in lines:
            arr = _as_vec2(pl)
            if arr is not None:
                ax.plot(arr[:, 0], arr[:, 1], color="#666666", lw=1.0, zorder=0)

    if isinstance(areas, (list, tuple)):
        for poly in areas:
            arr = _as_vec2(poly)
            if arr is not None and len(arr) >= 3:
                path = Path(arr, closed=True)
                patch = PathPatch(path, facecolor="#EFEFEF", edgecolor="#222222", lw=1.0)
                ax.add_patch(patch)

    # --- labels --------------------------------------------------------------
    patches: List[Tuple[int, Rectangle]] = []
    pos_arr = _as_vec2(pos)
    if pos_arr is None:
        pos_arr = np.zeros((0, 2))
    wh_arr = _as_vec2(rect_wh)
    if wh_arr is None:
        wh_arr = np.zeros_like(pos_arr)

    for i in range(min(len(labels), len(pos_arr), len(wh_arr))):
        x, y = pos_arr[i]
        w, h = wh_arr[i]
        rect = Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            facecolor=(0.95, 0.95, 1.0, 0.8),
            edgecolor="#2B6CB0",
            lw=1.0,
            picker=True,
            zorder=5,
        )
        ax.add_patch(rect)
        patches.append((i, rect))

        ax.text(x, y, _label_text(labels[i], i), ha="center", va="center", fontsize=9, zorder=6)

        if anchors is not None and i < len(anchors):
            ax.plot([anchors[i, 0], x], [anchors[i, 1], y], color="#888888", lw=1.0)
            circ = Circle(
                (anchors[i, 0], anchors[i, 1]),
                radius=5.0,
                facecolor=(0.1, 0.5, 0.9, 0.25),
                edgecolor="#2B6CB0",
                lw=1.0,
                zorder=4,
            )
            ax.add_patch(circ)

    return patches


def draw_force_panel(
    ax: plt.Axes,
    forces: Dict[str, np.ndarray],
    idx: int,
    title: str | None = None,
) -> Tuple[float, float]:
    """Draw a simple force decomposition diagram.

    Parameters
    ----------
    ax:
        Axis to draw on.
    forces:
        Mapping of component name to ``(N, 2)`` arrays.
    idx:
        Index of the currently selected label.
    title:
        Optional title for the panel, typically the label identifier.

    Returns
    -------
    Tuple[float, float]
        Magnitude and angle (degrees) of the total force acting on the
        selected label.
    """

    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0, color="#dddddd", lw=1.0)
    ax.axvline(0, color="#dddddd", lw=1.0)
    if title:
        ax.set_title(title, fontsize=10)

    vecs: List[Tuple[str, float, float]] = []
    for name, arr in forces.items():
        a = _as_vec2(arr)
        if a is not None and idx < len(a):
            vx, vy = float(a[idx, 0]), float(a[idx, 1])
            vecs.append((name, vx, vy))

    if not vecs:
        ax.text(0.5, 0.5, "no forces", transform=ax.transAxes, ha="center", va="center")
        return 0.0, 0.0

    vmax = max(np.hypot(vx, vy) for _, vx, vy in vecs)
    limit = vmax * 1.2 if vmax > 0 else 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")

    total_x = 0.0
    total_y = 0.0
    for name, vx, vy in vecs:
        color = FORCE_COLORS.get(name.lower(), "#777777")
        arr = FancyArrowPatch(
            (0, 0),
            (vx, vy),
            arrowstyle="->",
            mutation_scale=15,
            color=color,
            lw=2.0,
        )
        ax.add_patch(arr)
        ax.text(vx, vy, name, color=color, fontsize=6)
        total_x += vx
        total_y += vy

    total_arrow = FancyArrowPatch(
        (0, 0),
        (total_x, total_y),
        arrowstyle="-|>",
        mutation_scale=18,
        color=FORCE_COLORS["total"],
        lw=2.5,
    )
    ax.add_patch(total_arrow)

    mag = float(np.hypot(total_x, total_y))
    ang = float(np.degrees(np.arctan2(total_y, total_x)))
    return mag, ang


def draw_info_panel(
    ax: plt.Axes,
    forces: Dict[str, np.ndarray],
    idx: int,
    label_total: Tuple[float, float],
    global_total: Tuple[float, float],
    d_force: Optional[Tuple[float, float]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a textual summary of forces and optional metrics.

    Parameters
    ----------
    ax:
        Matplotlib axis used for drawing.
    forces:
        Mapping of force component names to ``(N, 2)`` arrays.
    idx:
        Index of the currently selected label.
    label_total:
        ``(|F|, angle)`` of the net force acting on the selected label.
    global_total:
        ``(|F|, angle)`` of the net force summed over **all** labels.
    d_force:
        Optional ``(ΔF, ΔF/F)`` pair describing the change of the global force
        compared to the previous iteration.  Values are ignored when ``None``.
    metrics:
        Optional optimiser statistics provided by ``metrics_getter``.
    """

    rows: List[Tuple[str, str, int]] = []
    if metrics:
        rows.append(
            (
                "iter={iter}  f={f:.3e}  |g|={gnorm:.2e}".format(
                    iter=metrics.get("iter", "?"),
                    f=float(metrics.get("f", np.nan)),
                    gnorm=float(metrics.get("gnorm_inf", np.nan)),
                ),
                "#222222",
                9,
            )
        )

    g_mag, g_ang = global_total
    rows.append(
        (f"ALL  |F|={g_mag:.3e}  angle={g_ang:+.1f}°", FORCE_COLORS["total"], 9)
    )
    if d_force is not None:
        d_abs, d_rel = d_force
        rows.append(
            (
                f"ΔF={d_abs:.3e}  ΔF/F={d_rel:.3e}",
                "#333333",
                9,
            )
        )

    l_mag, l_ang = label_total
    rows.append(
        (f"LABEL|F|={l_mag:.3e}  angle={l_ang:+.1f}°", FORCE_COLORS["total"], 9)
    )

    tot_mag = l_mag if l_mag > 0 else 1.0
    for name, arr in forces.items():
        a = _as_vec2(arr)
        if a is None or idx >= len(a):
            continue
        vx, vy = float(a[idx, 0]), float(a[idx, 1])
        comp_mag = float(np.hypot(vx, vy))
        comp_ang = float(np.degrees(np.arctan2(vy, vx)))
        pct = comp_mag / tot_mag * 100.0
        rows.append(
            (
                f"{name:<12s} |F|={comp_mag:.3e}  angle={comp_ang:+.1f}°  {pct:5.1f}%",
                FORCE_COLORS.get(name.lower(), "#555555"),
                8,
            )
        )

    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    for i, (text, color, size) in enumerate(rows):
        ax.text(
            0.01,
            0.99 - i * 0.075,
            text,
            ha="left",
            va="top",
            color=color,
            family="monospace",
            fontsize=size,
            transform=ax.transAxes,
        )


def draw_field_panel(
    ax: plt.Axes,
    field: Any,
    kind: str = "heatmap",
    cmap: str = "viridis",
) -> None:
    """Render a scalar field either as a heatmap or as a 3‑D surface.

    Parameters
    ----------
    ax:
        Matplotlib axis used for drawing.
    field:
        Two dimensional array containing the scalar values.
    kind:
        ``"heatmap"`` for a 2‑D colour map or ``"3d"`` for a surface plot.
    cmap:
        Name of the Matplotlib colormap to use.  Defaults to ``"viridis"``.
    """

    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    arr = None if field is None else np.asarray(field, dtype=float)
    if arr is None or arr.ndim != 2:
        ax.text(0.5, 0.5, "no field", ha="center", va="center", transform=ax.transAxes)
        return

    if kind == "3d":
        yy = np.arange(arr.shape[0])
        xx = np.arange(arr.shape[1])
        X, Y = np.meshgrid(xx, yy)
        ax.plot_surface(X, Y, arr, cmap=cmap)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.imshow(arr, origin="upper", cmap=cmap, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])

