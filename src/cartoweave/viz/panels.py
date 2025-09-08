"""Utility routines for simple visualisation panels.

This module provides small helpers used by :mod:`cartoweave.viz.view`.
The functions deliberately implement only the features required by the
examples and prioritise clarity over completeness.  They accept plain data
structures (mostly ``numpy`` arrays and ``dict`` objects) and draw using
Matplotlib primitives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path

from ..utils.layout_mode import is_circle_label
from .layout_style import LayoutStyle
from .metrics import collect_solver_metrics

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

    kind = getattr(lab, "kind", None)
    lab_id = getattr(lab, "id", None)
    if isinstance(lab, dict):
        kind = lab.get("kind", kind)
        lab_id = lab.get("id", lab_id)
    if kind is None:
        kind = "L"
    if lab_id is None:
        lab_id = index
    return f"{kind[:1]}{lab_id}"


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


def _fmt_sci3(x: float) -> str:
    """Format ``x`` using scientific notation with three significant digits."""

    return f"{float(x):.3e}"


def _fmt_force(x: float) -> str:
    """Format a force magnitude."""

    return _fmt_sci3(x)


def _fmt_deg_aligned(deg: float) -> str:
    """Format an angle in degrees with alignment."""

    return f"{float(deg):6.1f}°"


def _fmt_pct_aligned(pct: float) -> str:
    """Format a percentage with alignment."""

    return f"{float(pct):6.1f}%"


def _compose_info_rows(
    step: int,
    comp_dict: Dict[str, np.ndarray],
    idx: int,
    label_tot: tuple[float, float],
    global_tot: tuple[float, float],
    d_pair: Optional[tuple[float, float]],
    label_name: str,
    metrics: Dict[str, Any],
    force_colors: Dict[str, str],
    global_fs: int,
    label_fs: int,
    comp_fs: int,
) -> List[tuple[str, str, int, bool]]:
    """Compose text rows describing solver and force information."""

    rows: List[tuple[str, str, int, bool]] = []

    rows.append((f"frame={step}", "#000000", global_fs, False))

    ui = metrics.get("ui", {}) if isinstance(metrics, dict) else {}
    F_inf = float(metrics.get("F_inf", float("nan")))
    F_tol = float(metrics.get("F_tol", float("nan")))
    g_inf = float(metrics.get("g_inf", float("nan")))
    g_tol = float(metrics.get("gtol", float("nan")))
    F_style = ui.get("F_style", {"color": "#000000", "bold": False})
    g_style = ui.get("g_style", {"color": "#000000", "bold": False})

    r = float("inf")
    if np.isfinite(g_inf) and abs(g_inf) > 0:
        r = F_inf / g_inf
    r_disp = "Inf" if not np.isfinite(r) or r >= 100 else f"{r:.0f}"

    # Solver iteration info if available
    iter_cur = metrics.get("iter")
    iter_max = metrics.get("iter_max")
    solver_name = metrics.get("solver")
    if iter_cur is not None:
        if iter_max is not None:
            s = f"iter={int(iter_cur)}/{int(iter_max)}"
        else:
            s = f"iter={int(iter_cur)}"
        if solver_name:
            s = f"{solver_name} {s}"
        rows.append((s, "#000000", global_fs, False))

    rows.append(
        (
            f"F_inf={_fmt_sci3(F_inf)} tol={_fmt_sci3(F_tol)}",
            F_style.get("color", "#000000"),
            global_fs,
            bool(F_style.get("bold", False)),
        )
    )
    rows.append(
        (
            f"g_inf={_fmt_sci3(g_inf)} tol={_fmt_sci3(g_tol)} r={r_disp}",
            g_style.get("color", "#000000"),
            global_fs,
            bool(g_style.get("bold", False)),
        )
    )

    g_mag, g_ang = global_tot
    rows.append(
        (
            f"{'ALL'.ljust(6)} |F|={_fmt_force(g_mag)} angle={_fmt_deg_aligned(g_ang)}",
            "#000000",
            global_fs,
            False,
        )
    )
    if d_pair is not None:
        d_abs, d_rel = d_pair
        rows.append(
            (
                f"\u0394F={_fmt_sci3(d_abs)} \u0394F/F={float(d_rel):.2e}",
                "#000000",
                global_fs,
                False,
            )
        )

    l_mag, l_ang = label_tot
    name_fmt = label_name[:6].ljust(6)
    rows.append(
        (
            f"{name_fmt} |F|={_fmt_force(l_mag)} angle={_fmt_deg_aligned(l_ang)}",
            "#000000",
            label_fs,
            False,
        )
    )

    tot_mag = l_mag if l_mag > 0 else 1.0
    keys = [k for k in ALL_FORCE_KEYS if _as_vec2(comp_dict.get(k)) is not None]
    extras = [k for k, v in comp_dict.items() if _as_vec2(v) is not None and k not in keys]
    keys.extend(extras)
    if keys:
        rows.append(("components:", "#000000", label_fs, False))
        for k in keys:
            v = _as_vec2(comp_dict.get(k))
            if v is None or idx >= len(v):
                continue
            vx, vy = float(v[idx, 0]), float(v[idx, 1])
            mag = float(np.hypot(vx, vy))
            ang = float(np.degrees(np.arctan2(vy, vx)))
            pct = mag / tot_mag * 100.0
            name_fmt = k[:12].ljust(12)
            rows.append(
                (
                    f"  {name_fmt} |F|={_fmt_force(mag)} angle={_fmt_deg_aligned(ang)} {_fmt_pct_aligned(pct)}",
                    _force_color(k, force_colors),
                    comp_fs,
                    False,
                )
            )
    return rows


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

def _draw_layout_panel(
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
    style: LayoutStyle,
) -> List[Tuple[int, plt.Artist]]:
    """Render the main layout panel.

    The function returns a list of ``(index, Artist)`` tuples.  Patches are
    marked as pickable so that callers can detect clicks and change the selected
    label.
    """

    ax.clear()
    if style.equal_aspect:
        ax.set_aspect("equal")
    ax.set_xlim(0, frame_w)
    if style.invert_y:
        ax.set_ylim(frame_h, 0)
    else:
        ax.set_ylim(0, frame_h)
    ax.set_facecolor(style.background)

    # add simple tick marks to help gauge scale
    ax.set_xticks(np.linspace(0, frame_w, 5))
    ax.set_yticks(np.linspace(0, frame_h, 5))
    ax.grid(style.show_grid, color=style.grid_color, lw=style.grid_lw)

    # --- background geometry -------------------------------------------------

    pts = _as_vec2(points)
    if pts is not None:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            marker="x",
            c=style.points_color,
            s=style.point_size,
            zorder=1,
            linewidths=style.line_width,
        )

    if isinstance(lines, (list, tuple)):
        for pl in lines:
            arr = _as_vec2(pl)
            if arr is not None:
                ax.plot(
                    arr[:, 0],
                    arr[:, 1],
                    color=style.lines_color,
                    lw=style.line_width,
                    zorder=0,
                )

    if isinstance(areas, (list, tuple)):
        for poly in areas:
            arr = _as_vec2(poly)
            if arr is not None and len(arr) >= 3:
                path = Path(arr, closed=True)
                patch = PathPatch(
                    path,
                    facecolor=to_rgba(
                        style.areas_face_color, alpha=style.areas_face_alpha
                    ),
                    edgecolor=style.areas_edge_color,
                    lw=style.areas_edge_width,
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
        anchor_xy = None
        if anchors is not None and i < len(anchors):
            ax_, ay_ = anchors[i, 0], anchors[i, 1]
            if np.isfinite(ax_) and np.isfinite(ay_):
                anchor_xy = (float(ax_), float(ay_))
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
            if anchor_xy is None:
                continue
            x, y = anchor_xy
            radius = float(min(w, h)) / 2.0
            circ = Circle(
                (x, y),
                radius=radius,
                facecolor=style.labels_fill_color,
                edgecolor=style.labels_edge_color,
                lw=style.label_edge_width,
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
                facecolor=style.labels_fill_color,
                edgecolor=style.labels_edge_color,
                lw=style.label_edge_width,
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
                fontsize=style.label_fontsize,
                zorder=6,
            )

        if anchor_xy is not None:
            ax.plot(
                [anchor_xy[0], x],
                [anchor_xy[1], y],
                color=style.anchor_line_color,
                lw=style.anchor_line_width,
                linestyle='--',
            )
            ax.scatter(
                anchor_xy[0],
                anchor_xy[1],
                marker='x',
                c=style.anchor_marker_edge,
                s=(style.anchor_marker_size * 2) ** 2,
                linewidths=style.line_width,
                zorder=4,
            )

    return patches


def draw_layout(ax: plt.Axes, *args, style: LayoutStyle, **kwargs):
    """Dispatch to draw the layout panel.

    Supports two calling conventions:

    1. ``draw_layout(ax, pos, labels, rect_wh, *, frame_w, frame_h, ...)`` or
       the same with keyword arguments ``pos=...``, ``labels=...``, etc.
    2. ``draw_layout(ax, view_pack, t, style=style)``.
    """

    if args and isinstance(args[0], np.ndarray):
        return _draw_layout_panel(ax, *args, style=style, **kwargs)

    if "pos" in kwargs:
        pos = kwargs.pop("pos")
        labels = kwargs.pop("labels")
        rect_wh = kwargs.pop("rect_wh")
        return _draw_layout_panel(ax, pos, labels, rect_wh, style=style, **kwargs)

    if not args:
        raise TypeError("draw_layout expected ViewPack or position array")

    view_pack = args[0]
    if len(args) < 2:
        raise TypeError("draw_layout expected time index")
    t = int(args[1])

    fr = view_pack.frames[t]
    src = getattr(view_pack, "sources", None)
    frame_w = getattr(src, "frame_size", (1.0, 1.0))[0]
    frame_h = getattr(src, "frame_size", (1.0, 1.0))[1]
    points = getattr(src, "points", None)
    lines = getattr(src, "lines", None)
    areas_dict = getattr(src, "areas", None)
    areas = [
        a.get('xy')
        for a in areas_dict
        if isinstance(a, dict) and 'xy' in a and a['xy'] is not None
    ]
    rect_wh = getattr(view_pack, "WH", None)
    anchors = getattr(fr, "anchors", None)

    return _draw_layout_panel(
        ax,
        fr.P,
        getattr(view_pack, "labels", []),
        rect_wh if rect_wh is not None else np.zeros_like(fr.P),
        frame_w=float(frame_w),
        frame_h=float(frame_h),
        points=points,
        lines=lines,
        areas=areas,
        anchors=anchors,
        style=style,
    )


def draw_force_panel(
    ax: plt.Axes,
    forces: Dict[str, np.ndarray],
    idx: int,
    title: str | None = None,
    *,
    terms_to_plot: Optional[Sequence[str]] = None,
    viz_forces: Mapping[str, Any],
    viz_info: Mapping[str, Any],
    scale: float = 1.0,
    invert_y: bool = True,
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
            # Some force providers may already include a pre-summed total.
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
    limit = vmax * scale * 1.2 if vmax > 0 else 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")

    cfg = viz_forces

    total_x = sum(v[0] for v in vecs.values())
    total_y = sum(v[1] for v in vecs.values())

    terms = list(vecs.keys()) if terms_to_plot is None else [t for t in terms_to_plot if t in vecs]
    for name in terms:
        vx, vy, mag = vecs[name]
        vx *= scale
        vy *= scale
        plot_y = -vy if invert_y else vy
        color = _force_color(name, cfg["colors"])
        arr = FancyArrowPatch(
            (0, 0),
            (vx, plot_y),
            arrowstyle="->",
            mutation_scale=cfg["component_arrow_scale"],
            facecolor=color,
            edgecolor=color,
            lw=cfg["component_arrow_lw"],
        )
        ax.add_patch(arr)
        ax.text(vx, plot_y, name, color=color, fontsize=cfg["component_fontsize"])

    total_x_draw = total_x * scale
    total_y_draw = total_y * scale
    plot_total_y = -total_y_draw if invert_y else total_y_draw
    total_color = cfg.get("total_arrow_color", _force_color("total", cfg["colors"]))
    total_arrow = FancyArrowPatch(
        (0, 0),
        (total_x_draw, plot_total_y),
        arrowstyle="-|>",
        mutation_scale=cfg["total_arrow_scale"],
        facecolor=total_color,
        edgecolor=total_color,
        lw=cfg["total_arrow_lw"],
    )
    ax.add_patch(total_arrow)

    mag = float(np.hypot(total_x, total_y))
    ang = float(np.degrees(np.arctan2(total_y, total_x)))
    return mag, ang


def draw_field_panel(
    ax: plt.Axes,
    field: Any,
    width: float,
    height: float,
    kind: str = "3d",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = False,
) -> None:
    """Render a scalar field either as a heatmap or as a 3-D surface.

    Parameters
    ----------
    ax:
        Matplotlib axis used for drawing.
    field:
        2D array of scalar values.
    width, height:
        Frame dimensions.
    kind:
        "heatmap" | "3d" | "none".
    cmap:
        Matplotlib colormap name.
    vmin, vmax:
        Optional color scaling limits for consistent comparisons across frames.
    add_colorbar:
        If True, add a colorbar for the artist drawn.
    """
    ax.cla()
    arr = None if field is None else np.asarray(field, dtype=float)

    # 空内容：保留坐标框架，避免“看起来像没渲染”
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

    # 归一化器（同时适配 heatmap 和 3D surface）
    norm = None
    if (vmin is not None) or (vmax is not None):
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if kind == "3d":
        ny, nx = arr.shape
        xs = np.linspace(0.0, float(width), nx)
        ys = np.linspace(0.0, float(height), ny)
        X, Y = np.meshgrid(xs, ys)

        # 注意：plot_surface 支持 norm/cmap（通过 ScalarMappable）
        surf = ax.plot_surface(X, Y, arr, cmap=cmap, norm=norm, antialiased=True)

        ax.set_xlim(0, float(width))
        ax.set_ylim(float(height), 0)  # y 轴向下与 2D 视图一致
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("value")

        z_min = float(np.min(arr))
        z_max = float(np.max(arr))
        if z_max - z_min <= 0:
            z_max = z_min + 1.0
        ax.set_zlim(z_min, z_max)
        short_side = min(float(width), float(height))
        ax.set_box_aspect((float(width), float(height), short_side * (2.0 / 3.0)))

        if add_colorbar:
            ax.figure.colorbar(surf, ax=ax, fraction=0.046, pad=0.04)

    else:  # heatmap
        extent = grid_extent(width, height)
        im = ax.imshow(
            arr, origin="upper", cmap=cmap, extent=extent, aspect="auto", norm=norm
        )
        ax.set_xlim(0, float(width))
        ax.set_ylim(float(height), 0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks([])
        ax.set_yticks([])

        if add_colorbar:
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_forces(
    ax,
    view_pack,
    t: int,
    viz_cfg: dict,
    selected_idx: Optional[int] = None,
) -> None:
    """Render force vectors for the selected label."""

    frames = getattr(view_pack, "frames", [])
    labels = getattr(view_pack, "labels", [])
    if not frames:
        return
    if t < 0 or t >= len(frames):
        t = 0
    fr = frames[t]

    N = getattr(view_pack, "N", len(fr.P))
    comps_full = normalize_comps_for_info(fr.comps, N)
    forces_total = np.zeros((N, 2), dtype=float)
    for arr in comps_full.values():
        forces_total += np.asarray(arr, float)

    metrics = collect_solver_metrics(
        fr.P,
        forces_total,
        comps_full,
        labels,
        fr.metrics or {},
        viz_cfg,
    )

    top = metrics.get("top_force_labels") or [(0, 0.0)]
    if selected_idx is not None and 0 <= selected_idx < N:
        idx = int(selected_idx)
    else:
        idx = int(top[0][0]) if top else 0
    label_name = _label_text(labels[idx] if idx < len(labels) else {}, idx)

    forces_cfg = viz_cfg.get("forces", {})
    terms = select_terms_for_arrows(comps_full, forces_cfg)
    draw_force_panel(
        ax,
        comps_full,
        idx,
        label_name,
        terms_to_plot=terms,
        viz_forces=forces_cfg,
        viz_info=viz_cfg.get("info", {}),
        scale=float(forces_cfg.get("vector_scale", 1.0)),
        invert_y=True,
    )

    ax.set_title("forces")


def draw_info(
    ax,
    view_pack,
    t: int,
    viz_cfg: dict,
    selected_idx: Optional[int] = None,
) -> None:
    """Render the textual information panel."""

    ax.set_title("info", pad=8)
    ax.set_axis_off()

    frames = getattr(view_pack, "frames", [])
    labels = getattr(view_pack, "labels", [])
    if not frames:
        return
    if t < 0 or t >= len(frames):
        t = 0
    fr = frames[t]

    N = getattr(view_pack, "N", len(fr.P))
    comps_full = normalize_comps_for_info(fr.comps, N)
    forces_total = np.zeros((N, 2), dtype=float)
    for arr in comps_full.values():
        forces_total += np.asarray(arr, float)

    metrics = collect_solver_metrics(
        fr.P,
        forces_total,
        comps_full,
        labels,
        fr.metrics or {},
        viz_cfg,
    )

    top = metrics.get("top_force_labels") or [(0, 0.0)]
    if selected_idx is not None and 0 <= selected_idx < N:
        idx = int(selected_idx)
    else:
        idx = int(top[0][0]) if top else 0
    label_name = _label_text(labels[idx] if idx < len(labels) else {}, idx)

    l_vec = forces_total[idx]
    l_mag = float(np.hypot(l_vec[0], l_vec[1]))
    l_ang = float(np.degrees(np.arctan2(l_vec[1], l_vec[0])))
    g_vec = forces_total.sum(axis=0)
    g_mag = float(np.hypot(g_vec[0], g_vec[1]))
    g_ang = float(np.degrees(np.arctan2(g_vec[1], g_vec[0])))

    d_pair = None
    dF = metrics.get("deltaF")
    dF_rel = metrics.get("deltaF_over_F")
    if isinstance(dF, np.ndarray) and isinstance(dF_rel, np.ndarray):
        if idx < len(dF):
            d_pair = (float(dF[idx]), float(dF_rel[idx]))

    info_cfg = viz_cfg.get("info", {})
    forces_cfg = viz_cfg.get("forces", {})
    main_fs = int(info_cfg.get("row_main_fontsize", 12))
    comp_fs = int(info_cfg.get("row_component_fontsize", 10))
    rows = _compose_info_rows(
        t,
        comps_full,
        idx,
        (l_mag, l_ang),
        (g_mag, g_ang),
        d_pair,
        label_name,
        metrics,
        forces_cfg.get("colors", {}),
        main_fs,
        main_fs,
        comp_fs,
    )

    y = 0.98
    dy = 0.05
    for text, color, fs, bold in rows:
        ax.text(
            0.02,
            y,
            text,
            color=color,
            fontsize=fs,
            family="monospace",
            fontweight="bold" if bold else None,
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        y -= dy


def draw_field(ax, view_pack, t: int, viz_cfg: dict) -> None:
    """Render the scalar potential field for frame ``t``."""

    frames = getattr(view_pack, "frames", [])
    src = getattr(view_pack, "sources", None)
    if not frames or src is None:
        return
    if t < 0 or t >= len(frames):
        t = 0
    fr = frames[t]

    width, height = getattr(src, "frame_size", (1.0, 1.0))
    field_cfg = viz_cfg.get("field", {})

    kind_cfg = str(field_cfg.get("mode", "heatmap")).lower()
    kind = "3d" if "3d" in kind_cfg else "heatmap"
    cmap = str(field_cfg.get("cmap", "viridis"))
    res_short = int(field_cfg.get("resolution", 128)) or 128
    sigma = float(field_cfg.get("sigma", 1.2))  # 高斯核半径（以网格像素计）
    vmax = field_cfg.get("vmax", None)

    width = float(width)
    height = float(height)
    if width >= height:
        res_y = res_short
        res_x = max(1, int(round(res_short * width / max(height, 1e-9))))
    else:
        res_x = res_short
        res_y = max(1, int(round(res_short * height / max(width, 1e-9))))

    # 若已预计算，直接画
    field = getattr(fr, "field", None)
    if field is not None:
        draw_field_panel(ax, np.asarray(field, float), width, height, kind, cmap, vmax=vmax)
        ax.set_title("field")
        return

    # 1) 从 comps 聚合出总力（N,2）
    comps = getattr(fr, "comps", {}) or {}
    if not comps:
        empty = np.zeros((res_y, res_x), float)
        draw_field_panel(ax, empty, width, height, kind, cmap, vmax=vmax)
        ax.set_title("field (empty)")
        return

    first = next(iter(comps.values()))
    totalF = np.zeros_like(np.asarray(first, float))
    for arr in comps.values():
        totalF += np.asarray(arr, float)
    P = np.asarray(fr.P, float)
    assert P.shape[0] == totalF.shape[0] and totalF.shape[1] == 2, "P and force shape mismatch"

    # 2) 规则网格
    xs = np.linspace(0.0, width, res_x, endpoint=False)
    ys = np.linspace(0.0, height, res_y, endpoint=False)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # 3) 将散点力 splat 到网格（双线性 + 可选高斯）
    Fx = np.zeros((res_y, res_x), float)
    Fy = np.zeros((res_y, res_x), float)

    gx = np.clip(np.floor((P[:, 0] / max(width, 1e-9)) * res_x).astype(int), 0, res_x - 1)
    gy = np.clip(np.floor((P[:, 1] / max(height, 1e-9)) * res_y).astype(int), 0, res_y - 1)
    for (ix, iy, f) in zip(gx, gy, totalF):
        Fx[iy, ix] += float(f[0])
        Fy[iy, ix] += float(f[1])

    if sigma > 0:
        rad = int(max(1, round(3 * sigma)))
        k = np.arange(-rad, rad + 1, dtype=float)
        g = np.exp(-0.5 * (k / sigma) ** 2)
        g /= g.sum()

        Fx = np.apply_along_axis(lambda v: np.convolve(v, g, mode="same"), axis=1, arr=Fx)
        Fy = np.apply_along_axis(lambda v: np.convolve(v, g, mode="same"), axis=1, arr=Fy)
        Fx = np.apply_along_axis(lambda v: np.convolve(v, g, mode="same"), axis=0, arr=Fx)
        Fy = np.apply_along_axis(lambda v: np.convolve(v, g, mode="same"), axis=0, arr=Fy)

    # 4) 计算散度 div(F) ≈ dFx/dx + dFy/dy（中心差分，周期边界）
    dFx_dx = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1)) / (2 * dx)
    dFy_dy = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0)) / (2 * dy)
    divF = dFx_dx + dFy_dy

    # 5) 解 Poisson: ∇²φ = divF，频域解（周期 BC）
    div_hat = np.fft.rfft2(divF)
    ky = 2 * np.pi * np.fft.fftfreq(res_y, d=dy)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(res_x, d=dx)[None, :]
    denom = (kx ** 2 + ky ** 2)
    denom[0, 0] = 1.0  # 避免除零，DC 分量设为 0 势基准
    phi_hat = -div_hat / denom
    phi_hat[0, 0] = 0.0  # 去除未定常数
    phi = np.fft.irfft2(phi_hat, s=divF.shape)

    # 6) 归一化（改善可视化）：零均值 / min-max
    phi = phi - np.mean(phi)
    if vmax is None:
        m = np.max(np.abs(phi)) or 1.0
        vmax_local = m
    else:
        vmax_local = vmax

    draw_field_panel(ax, phi, width, height, kind, cmap, vmax=vmax_local)
    ax.set_title("potential φ (∇²φ=div F)")
