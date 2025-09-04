"""Utility routines for simple visualisation panels.

This module provides small helpers used by :mod:`cartoweave.viz.view`.
The functions deliberately implement only the features required by the
examples and prioritise clarity over completeness.  They accept plain data
structures (mostly ``numpy`` arrays and ``dict`` objects) and draw using
Matplotlib primitives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from ..layout_utils import is_circle_label

# Configuration is supplied externally.  Functions accept the relevant slices of
# the viewer configuration so that callers can merge YAML defaults beforehand.

# ---------------------------------------------------------------------------
# Force term helpers
# ---------------------------------------------------------------------------

# Superset of force keys used for zero-filling and consistent ordering in the
# info panel.  Additional keys encountered at runtime are appended verbatim.
ALL_FORCE_KEYS: List[str] = [
    "ll.rect",
    "ll.disk",
    "boundary.wall",
    "pl.rect",
    "ln.rect",
    "area.embed",
    "area.cross",
    "area.softout",
    "focus.attract",
    "anchor.spring",
]


def normalize_comps_for_info(comps: Dict[str, np.ndarray], N: int) -> Dict[str, np.ndarray]:
    """Ensure that *comps* contains all known force keys.

    Missing entries are filled with zero arrays of shape ``(N, 2)`` so the info
    panel can list every term for debugging purposes.
    """

    out: Dict[str, np.ndarray] = {}
    for k in ALL_FORCE_KEYS:
        arr = comps.get(k)
        if arr is None:
            arr = np.zeros((N, 2), dtype=np.float64)
        else:
            arr = np.asarray(arr, dtype=np.float64)
        out[k] = arr
    for k, v in comps.items():
        if k not in out:
            out[k] = np.asarray(v, dtype=np.float64)
    return out


def select_terms_for_arrows(comps: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> List[str]:
    """Return force component names that should be drawn as arrows."""

    if cfg.get("arrows_show_all", False):
        return [k for k in comps.keys()]

    term_peak: Dict[str, float] = {}
    peak = 0.0
    for k, arr in comps.items():
        if arr is None:
            continue
        a = _as_vec2(arr)
        if a is None:
            continue
        mag = np.linalg.norm(a, axis=1)
        m = float(np.max(mag)) if mag.size else 0.0
        term_peak[k] = m
        peak = max(peak, m)

    keep: List[str] = []
    min_abs = float(cfg.get("arrows_min_abs", 1e-9))
    min_ratio = float(cfg.get("arrows_min_ratio", 1e-2))
    for k, m in term_peak.items():
        if m >= min_abs and (peak == 0.0 or (m / peak) >= min_ratio):
            keep.append(k)

    cap = cfg.get("arrows_max_terms")
    if isinstance(cap, int) and cap > 0 and len(keep) > cap:
        keep = sorted(keep, key=lambda x: term_peak[x], reverse=True)[:cap]

    return keep


def _force_color(name: str, colors: Dict[str, str]) -> str:
    """Return a colour for a force component ``name``.

    Unknown names are assigned colours from Matplotlib's default cycle so that
    each component still receives a distinct hue.  Colours are drawn from the
    caller-provided ``colors`` mapping which may be mutated to cache new
    assignments.
    """

    key = name.lower()
    if key not in colors:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if cycle:
            colors[key] = to_hex(cycle[len(colors) % len(cycle)], keep_alpha=True)
        else:
            colors[key] = "#777777FF"  # grey default
    return colors[key]


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


def _format_pct(pct: float) -> str:
    """Return a percentage string with aligned decimal places.

    The info panel lists force contributions as percentages.  To facilitate
    comparison across rows we format the values so that the decimal points line
    up vertically.  ``pct`` is expected to be given in ``[0, 100]`` units.
    """

    s = f"{pct:14.1f}"  # sign + four integer digits + decimal
    sign = s[0]
    integer = s[1:5].lstrip("0") or "0"
    integer = integer.rjust(4, " ")
    return sign + integer + s[5:] + "%"


def grid_extent(width: float, height: float) -> Tuple[float, float, float, float]:
    """Return ``imshow``/``pcolormesh`` extents for a grid.

    The returned tuple follows Matplotlib's ``extent`` order for
    ``origin='upper'``: ``(xmin, xmax, ymax, ymin)``.  Using a dedicated helper
    keeps the conversion in one place and avoids subtle axis flip bugs when
    different panels compute extents independently.
    """

    return (0.0, float(width), float(height), 0.0)


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
    viz_layout: Mapping[str, Any],
) -> List[Tuple[int, plt.Artist]]:
    """Render the main layout panel.

    The function returns a list of ``(index, Artist)`` tuples.  Patches are
    marked as pickable so that callers can detect clicks and change the selected
    label.
    """

    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(0, frame_w)
    ax.set_ylim(frame_h, 0)  # origin in the top-left corner
    ax.set_facecolor("#FFFFFFFF")  # white background

    # add simple tick marks to help gauge scale
    ax.set_xticks(np.linspace(0, frame_w, 5))
    ax.set_yticks(np.linspace(0, frame_h, 5))
    ax.grid(True, color="#DDDDDDFF", lw=0.5)  # light gray grid

    # --- background geometry -------------------------------------------------
    cfg = viz_layout
    colors = cfg["colors"]

    pts = _as_vec2(points)
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], c=colors["points"], s=18, zorder=1)

    if isinstance(lines, (list, tuple)):
        for pl in lines:
            arr = _as_vec2(pl)
            if arr is not None:
                ax.plot(
                    arr[:, 0],
                    arr[:, 1],
                    color=colors["lines"],
                    lw=cfg["line_width"],
                    zorder=0,
                )

    if isinstance(areas, (list, tuple)):
        for poly in areas:
            arr = _as_vec2(poly)
            if arr is not None and len(arr) >= 3:
                path = Path(arr, closed=True)
                patch = PathPatch(
                    path,
                    facecolor=to_rgba(colors["areas"], alpha=cfg["area_face_alpha"]),
                    edgecolor=colors["areas"],
                    lw=cfg["area_edge_width"],
                )
                ax.add_patch(patch)

    # --- labels --------------------------------------------------------------
    patches: List[Tuple[int, plt.Artist]] = []
    pos_arr = _as_vec2(pos)
    if pos_arr is None:
        pos_arr = np.zeros((0, 2))
    wh_arr = _as_vec2(rect_wh)
    if wh_arr is None:
        wh_arr = np.zeros_like(pos_arr)

    for i in range(min(len(labels), len(pos_arr), len(wh_arr))):
        x, y = pos_arr[i]
        w, h = wh_arr[i]
        if not (
            np.isfinite(x)
            and np.isfinite(y)
            and np.isfinite(w)
            and np.isfinite(h)
        ):
            continue
        if w <= 0 or h <= 0:
            continue
        lab = labels[i] if i < len(labels) else {}
        if is_circle_label(lab):
            radius = float(min(w, h)) / 2.0
            circ = Circle(
                (x, y),
                radius=radius,
                facecolor=colors.get("label_fill", "#F2F2FFCC"),
                edgecolor=colors.get("label_edge", "#2B6CB0FF"),
                lw=cfg["label_edge_width"],
                picker=True,
                zorder=5,
            )
            ax.add_patch(circ)
            patches.append((i, circ))
        else:
            rect = Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                facecolor=colors["label_fill"],
                edgecolor=colors["label_edge"],
                lw=cfg["label_edge_width"],
                picker=True,
                zorder=5,
            )
            ax.add_patch(rect)
            patches.append((i, rect))

            ax.text(
                x,
                y,
                _label_text(labels[i], i),
                ha="center",
                va="center",
                fontsize=cfg["label_fontsize"],
                zorder=6,
            )

        if ~np.isnan(anchors[i]).all() and i < len(anchors):
            ax.plot(
                [anchors[i, 0], x],
                [anchors[i, 1], y],
                color=colors["anchor_line"],
                lw=cfg["line_width"],
            )
            ax.scatter(anchors[i, 0], anchors[i, 1], marker='x', s=(cfg.get("anchor_marker_size", 4.0) * 2) ** 2,
                       c=colors["anchor_marker_edge"], lw=cfg["line_width"], zorder=4)

    return patches


def draw_force_panel(
    ax: plt.Axes,
    forces: Dict[str, np.ndarray],
    idx: int,
    title: str | None = None,
    *,
    terms_to_plot: Optional[Sequence[str]] = None,
    viz_forces: Mapping[str, Any],
    viz_info: Mapping[str, Any],
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
    ax.axhline(0, color="#DDDDDDFF", lw=1.0)  # light gray axes
    ax.axvline(0, color="#DDDDDDFF", lw=1.0)  # light gray axes
    if title:
        ax.set_title(title, fontsize=viz_info["title_fontsize"])

    vecs: Dict[str, Tuple[float, float, float]] = {}
    for name, arr in forces.items():
        if name == "total":
            # Some force providers may already include a pre‑summed total.
            continue
        a = _as_vec2(arr)
        if a is not None and idx < len(a):
            vx, vy = float(a[idx, 0]), float(a[idx, 1])
            vecs[name] = (vx, vy, float(np.hypot(vx, vy)))

    if not vecs:
        ax.text(0.5, 0.5, "no forces", transform=ax.transAxes, ha="center", va="center")
        return 0.0, 0.0

    mags = [v[2] for v in vecs.values()]
    vmax = max(mags)
    limit = vmax * 1.2 if vmax > 0 else 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")

    cfg = viz_forces

    total_x = sum(v[0] for v in vecs.values())
    total_y = sum(v[1] for v in vecs.values())

    terms = list(vecs.keys()) if terms_to_plot is None else [t for t in terms_to_plot if t in vecs]
    for name in terms:
        vx, vy, mag = vecs[name]
        color = _force_color(name, cfg["colors"])
        arr = FancyArrowPatch(
            (0, 0),
            (vx, vy),
            arrowstyle="->",
            mutation_scale=cfg["component_arrow_scale"],
            color=color,
            edgecolor=color,
            lw=cfg["component_arrow_lw"],
        )
        ax.add_patch(arr)
        ax.text(vx, vy, name, color=color, fontsize=cfg["component_fontsize"])

    total_arrow = FancyArrowPatch(
        (0, 0),
        (total_x, total_y),
        arrowstyle="-|>",
        mutation_scale=cfg["total_arrow_scale"],
        color=_force_color("total", cfg["colors"]),
        edgecolor=_force_color("total", cfg["colors"]),
        lw=cfg["total_arrow_lw"],
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
    *,
    viz_info: Mapping[str, Any],
    viz_forces: Mapping[str, Any],
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

    cfg = viz_info

    rows: List[Tuple[str, str, int]] = []
    if metrics:
        rows.append(
            (
                "iter={iter}  f={f:.3e}  |g|={gnorm:.2e}".format(
                    iter=metrics.get("iter", "?"),
                    f=float(metrics.get("f", np.nan)),
                    gnorm=float(metrics.get("gnorm_inf", np.nan)),
                ),
                "#222222FF",  # dark grey text
                cfg["row_main_fontsize"],
            )
        )

    g_mag, g_ang = global_total
    name_fmt = "ALL".ljust(6)
    rows.append(
        (
            f"{name_fmt} |F|={g_mag:+9.3e} angle={g_ang:+6.1f}°       ",
            _force_color("total", viz_forces["colors"]),
            cfg["row_main_fontsize"],
        )
    )
    if d_force is not None:
        d_abs, d_rel = d_force
        rows.append(
            (
                f"ΔF={d_abs:.3e}  ΔF/F={d_rel:.3e}",
                "#333333FF",  # medium dark grey text
                cfg["row_main_fontsize"],
            )
        )

    l_mag, l_ang = label_total
    name_fmt = "LABEL".ljust(6)
    rows.append(
        (
            f"{name_fmt} |F|={l_mag:+9.3e} angle={l_ang:+6.1f}°       ",
            _force_color("total", viz_forces["colors"]),
            cfg["row_main_fontsize"],
        )
    )

    tot_mag = l_mag if l_mag > 0 else 1.0
    for name in sorted(forces):
        arr = forces[name]
        a = _as_vec2(arr)
        if a is None or idx >= len(a):
            continue
        vx, vy = float(a[idx, 0]), float(a[idx, 1])
        comp_mag = float(np.hypot(vx, vy))
        comp_ang = float(np.degrees(np.arctan2(vy, vx)))
        pct = comp_mag / tot_mag * 100.0
        pct_txt = _format_pct(pct)
        name_fmt = name[:6].ljust(6)
        row_txt = (
            f"{name_fmt} |F|={comp_mag:+9.3e} angle={comp_ang:+6.1f}° {pct_txt}"
        )
        rows.append(
            (
                row_txt,
                _force_color(name, viz_forces["colors"]),
                cfg["row_component_fontsize"],
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
    width: float,
    height: float,
    kind: str = "3d",
    cmap: str = "viridis",
) -> None:
    """Render a scalar field either as a heatmap or as a 3‑D surface.

    Parameters
    ----------
    ax:
        Matplotlib axis used for drawing.
    field:
        Two dimensional array containing the scalar values.
    width, height:
        Dimensions of the frame the field corresponds to.
    kind:
        ``"heatmap"`` for a 2‑D colour map, ``"3d"`` for a surface plot or
        ``"none"`` for an empty panel.
    cmap:
        Name of the Matplotlib colormap to use.  Defaults to ``"viridis"``.
    """

    # ``Axes3D.clear`` resets the plot and also removes any previously set
    # axis limits.  When no field data is available Matplotlib would otherwise
    # draw an empty panel without any frame or axes which looks as if the plot
    # failed to render.  Explicitly reset sensible limits so that a coordinate
    # frame remains visible even for missing data.
    ax.cla()
    arr = None if field is None else np.asarray(field, dtype=float)
    if kind == "none" or arr is None or arr.ndim != 2:
        ax.set_xlim(0, float(width))
        ax.set_ylim(float(height), 0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if kind != "none":
            if getattr(ax, "name", "") == "3d":
                ax.set_zlim(0, 1)
                ax.set_zlabel("value")
                xy_scale = max(float(width), float(height), 1.0)
                ax.set_box_aspect((float(width), float(height), xy_scale))
                ax.text2D(0.5, 0.5, "no field", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, "no field", ha="center", va="center", transform=ax.transAxes)
        return

    if kind == "3d":
        ny, nx = arr.shape
        xs = np.linspace(0.0, float(width), nx)
        ys = np.linspace(0.0, float(height), ny)
        X, Y = np.meshgrid(xs, ys)
        ax.plot_surface(X, Y, arr, cmap=cmap)
        ax.set_xlim(0, float(width))
        ax.set_ylim(float(height), 0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("value")
        z_span = float(arr.max() - arr.min()) or 1.0
        xy_scale = max(float(width), float(height), 1.0)
        z_aspect = min(z_span, xy_scale)
        ax.set_box_aspect((float(width), float(height), z_aspect))
    else:
        extent = grid_extent(width, height)
        ax.imshow(arr, origin="upper", cmap=cmap, extent=extent, aspect="auto")
        ax.set_xlim(0, float(width))
        ax.set_ylim(float(height), 0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks([])
        ax.set_yticks([])

