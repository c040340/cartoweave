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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path

from ..compute.forces import term_params_map
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


def _active_anchor_indices(
    labels: Sequence[Any], active_mask: Optional[Sequence[bool]]
) -> dict[str, set[int]]:
    """Return geometry indices referenced by active labels.

    The returned mapping contains index sets for ``"point"``, ``"line"`` and
    ``"area"`` targets.  Only labels whose ``anchor`` targets one of these
    geometry types and are marked active in ``active_mask`` contribute to the
    result.  ``active_mask`` may be ``None`` meaning all labels are considered
    active.
    """

    mask = None if active_mask is None else np.asarray(active_mask, dtype=bool)
    out: dict[str, set[int]] = {"point": set(), "line": set(), "area": set()}
    for i, lab in enumerate(labels):
        anchor = getattr(lab, "anchor", None)
        if anchor is None and isinstance(lab, dict):
            anchor = lab.get("anchor")
        if anchor is None:
            continue
        target = getattr(anchor, "target", None)
        index = getattr(anchor, "index", None)
        if isinstance(anchor, dict):
            target = anchor.get("target", target)
            index = anchor.get("index", index)
        if target in out and isinstance(index, (int, np.integer)):
            if mask is None or (i < len(mask) and mask[i]):
                out[target].add(int(index))
    return out


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
    active_mask: Optional[Sequence[bool]] = None,
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

    mask = None if active_mask is None else np.asarray(active_mask, dtype=bool)
    geom_refs = _active_anchor_indices(labels, mask)

    # --- background geometry -------------------------------------------------

    pts = _as_vec2(points)
    if pts is not None and geom_refs["point"]:
        idx = [i for i in sorted(geom_refs["point"]) if 0 <= i < len(pts)]
        if idx:
            arr = pts[idx]
            ax.scatter(
                arr[:, 0],
                arr[:, 1],
                marker="x",
                c=style.points_color,
                s=style.point_size,
                zorder=1,
                linewidths=style.line_width,
            )

    if isinstance(lines, (list, tuple)) and geom_refs["line"]:
        keep = geom_refs["line"]
        for i, pl in enumerate(lines):
            if i not in keep:
                continue
            arr = _as_vec2(pl)
            if arr is not None:
                ax.plot(
                    arr[:, 0],
                    arr[:, 1],
                    color=style.lines_color,
                    lw=style.line_width,
                    zorder=0,
                )

    if isinstance(areas, (list, tuple)) and geom_refs["area"]:
        keep_a = geom_refs["area"]
        for i, poly in enumerate(areas):
            if i not in keep_a:
                continue
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
        if mask is not None and (i >= len(mask) or not mask[i]):
            continue
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
                lw=max(style.anchor_line_width, 1.0),
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
    rect_wh_global = getattr(view_pack, "WH", None)
    rect_wh = fr.meta.get("WH", rect_wh_global)
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
        active_mask=fr.active_mask,
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


def draw_field(ax, view_pack, t: int, viz_cfg: dict, _draw_field_splat=None) -> None:
    """Render a force field for frame ``t`` using term probes (robust)."""
    import logging

    import numpy as np

    # 延迟导入：与你工程的模块路径保持一致
    from cartoweave.compute.forces import REGISTRY, get_probe

    log = logging.getLogger(__name__)

    # ---------- 取帧 ----------
    frames = getattr(view_pack, "frames", [])
    if not frames:
        ax.set_title("Field: no frames")
        return
    if t < 0 or t >= len(frames):
        t = 0
    fr = frames[t]

    # ---------- 解析 viz 配置 ----------
    field_cfg = viz_cfg.get("field", {}) or {}
    terms_cfg = field_cfg.get("terms")
    res_short = int(field_cfg.get("resolution", 128)) or 128
    aggregate = str(field_cfg.get("aggregate", "norm")).lower()
    log_scale = bool(field_cfg.get("log_scale", False))
    fallback_splat = bool(field_cfg.get("fallback_splat", True))
    kind_cfg = str(field_cfg.get("mode", "heatmap")).lower()
    kind = "3d" if "3d" in kind_cfg else "heatmap"
    cmap = str(field_cfg.get("cmap", "viridis"))
    vmax = field_cfg.get("vmax", None)
    extent = field_cfg.get("extent")

    # ---------- frame size / extent ----------
    # 优先顺序：viz.extent → view_pack.frame_size → fr.meta.frame_size → sources.frame_size → 兜底
    width = height = None
    xmin = ymin = 0.0
    xmax = ymax = None

    if extent and len(extent) == 4:
        xmin, xmax, ymin, ymax = map(float, extent)
        width = xmax - xmin
        height = ymax - ymin
    else:
        fs = getattr(view_pack, "frame_size", None)
        if isinstance(fs, (list, tuple)) and len(fs) == 2:
            width, height = float(fs[0]), float(fs[1])
        else:
            meta = getattr(fr, "meta", {}) or {}
            fs = meta.get("frame_size")
            if fs and len(fs) == 2:
                width, height = float(fs[0]), float(fs[1])
            else:
                src = getattr(view_pack, "sources", None)
                if src is not None and hasattr(src, "frame_size"):
                    fs = getattr(src, "frame_size", (1000.0, 1000.0))
                    width, height = float(fs[0]), float(fs[1])
                else:
                    width, height = 1000.0, 1000.0
        xmin, ymin = 0.0, 0.0
        xmax, ymax = width, height

    width = float(max(width, 1e-9))
    height = float(max(height, 1e-9))

    # ---------- 生成采样网格 ----------
    if width >= height:
        res_y = res_short
        res_x = max(1, int(round(res_short * width / height)))
    else:
        res_x = res_short
        res_y = max(1, int(round(res_short * height / width)))

    xs = np.linspace(xmin, xmax, res_x, endpoint=False)
    ys = np.linspace(ymin, ymax, res_y, endpoint=False)
    XX, YY = np.meshgrid(xs, ys)
    XY = np.stack([XX.ravel(), YY.ravel()], axis=1)

    # ---------- scene 构造（不强依赖 sources） ----------
    src = getattr(view_pack, "sources", None)
    labels = getattr(view_pack, "labels", [])
    WH = getattr(view_pack, "WH", None)
    scene = {
        "frame_size": (width, height),
        "areas": getattr(src, "areas", []) if src is not None else [],
        "points": getattr(src, "points", np.zeros((0, 2))),
        "lines": getattr(src, "lines", []) if src is not None else [],
        "labels": labels,
        "WH": WH,
        "labels_xy": np.asarray(getattr(fr, "P", np.zeros((0, 2))), float),
    }

    # ---------- 规范化 terms ----------
    def _normalize_terms(ts) -> List[str]:
        # 默认从 comps keys 推断
        if not ts:
            ts = list((getattr(fr, "comps", {}) or {}).keys())
        out: List[str] = []
        for name in ts:
            name = str(name)
            if name in REGISTRY:
                out.append(name)
                continue
            if "." not in name:
                # 简写匹配：e.g. "focus" -> "focus.attract"
                cands = [k for k in REGISTRY.keys() if k.split(".", 1)[0] == name]
                if cands:
                    out.extend(cands)
                    continue
            # 前缀匹配兜底
            cands = [k for k in REGISTRY.keys() if k.startswith(name)]
            if cands:
                out.extend(cands)
            else:
                log.debug("Field: unknown term '%s' (skipped)", name)
        # 去重保序
        seen = set()
        uniq = []
        for k in out:
            if k not in seen:
                uniq.append(k)
                seen.add(k)
        return uniq

    terms = _normalize_terms(terms_cfg)

    # ---------- 参数映射（多源合并 + 安全默认） ----------
    # 优先 defaults.compute → view_pack.compute → fr.params → 安全默认
    compute_cfg_all: Dict = {}
    # defaults.compute
    defaults = getattr(view_pack, "defaults", None)
    if isinstance(defaults, dict):
        node = defaults.get("compute") or defaults
        if isinstance(node, dict):
            compute_cfg_all.update(node)
    # view_pack.compute（若存在）
    vp_compute = getattr(view_pack, "compute", None)
    if isinstance(vp_compute, dict):
        # 不破坏已有键，补充合并
        for k, v in vp_compute.items():
            if k not in compute_cfg_all:
                compute_cfg_all[k] = v

    # 帧级 params（有些实现把 term 参数直接挂在 fr.params 或 fr.params["forces"]）
    fr_params = getattr(fr, "params", None)
    if isinstance(fr_params, dict):
        # 合并进 forces
        forces_slot = compute_cfg_all.setdefault("forces", {})
        # 允许两种结构：扁平 term->params 或 分组 forces[group][name]
        # 若 fr_params 下直接是 term 名，就并入
        for k, v in fr_params.items():
            if isinstance(v, dict):
                # 若是分组，则再并一次
                if k in ("ll", "ln", "pl", "area", "anchor", "boundary", "focus"):
                    forces_slot.setdefault(k, {}).update(v)
                else:
                    # 认为是扁平
                    if k not in forces_slot:
                        forces_slot[k] = v

    # 如果项目里提供了 term_params_map，就优先使用；否则走我们自定义解析
    if term_params_map is not None:
        pmap = term_params_map(compute_cfg_all)
    else:
        # 自定义：支持 flat 和 group.name 两种
        forces = compute_cfg_all.get("forces", {}) or {}
        pmap: Dict[str, Dict] = {}
        # 扁平
        for k, v in forces.items():
            if isinstance(v, dict) and "." in k:
                pmap[k] = v
        # 分组
        for g, gdict in forces.items():
            if isinstance(gdict, dict):
                for n, v in gdict.items():
                    if isinstance(v, dict):
                        pmap[f"{g}.{n}"] = v

    def _params_for(term: str) -> Dict:
        p = dict(pmap.get(term, {}) or {})
        # 安全默认
        p.setdefault("enable", True)
        p.setdefault("k", float(field_cfg.get("k", 1.0)))
        return p

    # ---------- 累加 probe ----------
    F = np.zeros_like(XY)
    used_terms: List[str] = []
    for term in terms:
        probe = get_probe(term)
        if probe is None:
            continue
        params = _params_for(term)
        if not params.get("enable", True):
            log.debug("Field: term '%s' disabled by params", term)
            continue
        try:
            Fi = probe(scene, params, XY)
            Fi = np.asarray(Fi, float)
            if Fi.shape != XY.shape:
                raise ValueError(f"probe('{term}') returned shape {Fi.shape}, expected {XY.shape}")
            if not np.all(np.isfinite(Fi)):
                raise ValueError(f"probe('{term}') produced non-finite values")
            F += Fi
            used_terms.append(term)
        except Exception as e:
            log.warning("Field: probe('%s') failed: %s", term, e)

    # ---------- 渲染 ----------
    if used_terms:
        Fx = F[:, 0].reshape(res_y, res_x)
        Fy = F[:, 1].reshape(res_y, res_x)
        if aggregate == "div":
            dx = (xmax - xmin) / max(res_x, 1)
            dy = (ymax - ymin) / max(res_y, 1)
            dFx_dx = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1)) / (2 * dx)
            dFy_dy = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0)) / (2 * dy)
            field = dFx_dx + dFy_dy
            label = "div"
        else:
            field = np.sqrt(Fx * Fx + Fy * Fy)
            if log_scale:
                field = np.log1p(field)
            label = "|F|"
        draw_field_panel(ax, field, width, height, kind, cmap, vmax=vmax)
        ax.set_title(f"Field (probe): {' + '.join(used_terms)} | {label} | {res_x}x{res_y}")
        return

    # ---------- fallback: label-splat ----------
    if fallback_splat:
        try:
            P = np.asarray(fr.P, float)
            comps = getattr(fr, "comps", {}) or {}
            if comps and P.size:
                comps = normalize_comps_for_info(comps, P.shape[0])
                totalF = np.zeros_like(P)
                for arr in comps.values():
                    totalF += np.asarray(arr, float)
                Fx = np.zeros((res_y, res_x), float)
                Fy = np.zeros((res_y, res_x), float)
                gx = np.clip(
                    np.floor(((P[:, 0] - xmin) / max(width, 1e-9)) * res_x).astype(int),
                    0,
                    res_x - 1,
                )
                gy = np.clip(
                    np.floor(((P[:, 1] - ymin) / max(height, 1e-9)) * res_y).astype(int),
                    0,
                    res_y - 1,
                )
                for (ix, iy, f) in zip(gx, gy, totalF):
                    Fx[iy, ix] += float(f[0])
                    Fy[iy, ix] += float(f[1])
                field = np.linalg.norm(np.stack([Fx, Fy], axis=2), axis=2)
                if log_scale:
                    field = np.log1p(np.abs(field))
                draw_field_panel(ax, field, width, height, kind, cmap, vmax=vmax)
                ax.set_title(
                    f"Field (fallback: label-splat): {' + '.join(terms)} | {aggregate} | {res_x}x{res_y}"
                )
                return
        except Exception as e:
            log.warning("Field: fallback splat failed: %s", e)

    # ---------- 空场 ----------
    empty = np.zeros((res_y, res_x), float)
    draw_field_panel(ax, empty, width, height, kind, cmap, vmax=vmax)
    ax.set_title("Field (no-probe, no-fallback)")

