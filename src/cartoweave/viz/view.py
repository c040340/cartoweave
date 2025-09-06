"""High level interactive visualisation.

The original project uses a rather feature rich viewer implemented in a large
script.  For the purposes of the exercises in this repository we only require
something lightweight that exposes the essential ideas – a sequence of steps of
label positions, per‑label force breakdown and an optional scalar field
visualised as either a heatmap or a 3‑D surface.  This module intentionally
keeps the code compact and easy to follow.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from .panels import (
    ALL_FORCE_KEYS,
    draw_field_panel,
    draw_force_panel,
    draw_layout,
    normalize_comps_for_info,
    select_terms_for_arrows,
)
from .defaults import merge_defaults
from cartoweave.utils.logging import logger
from ..labels import anchor_xy
import warnings


def _as_vec2(a: Any) -> Optional[np.ndarray]:
    """Return ``a`` as ``(N, 2)`` float array or ``None``."""

    if a is None:
        return None
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    return None


def _assert_vec2(arr: Any, name: str) -> np.ndarray:
    """Assert that ``arr`` is a ``(N,2)`` float array."""
    arr = np.asarray(arr, float)
    if not (arr.ndim == 2 and arr.shape[1] == 2):
        raise ValueError(f"{name} must be (N,2), got shape={arr.shape}")
    return arr


def _compute_anchors(
    labels: Sequence[Dict[str, Any]],
    *,
    points: Any = None,
    lines: Any = None,
    areas: Any = None,
    sources: Optional[Dict[str, Any]] = None,
    frame_size: Optional[tuple[float, float]] = None,
) -> Optional[np.ndarray]:
    """Find anchor positions for ``labels``.

    ``sources`` may already contain anchor coordinates under the key
    ``"anchors_xy"`` or ``"anchor_xy"``.  When not provided, the anchors are
    inferred from the background geometry based on ``anchor_kind`` and
    ``anchor_index`` stored inside each label.  Only a few common formats are
    supported and any missing information results in ``NaN`` anchors.

    ``frame_size`` defaults to ``(0, 0)`` which previously caused all anchors to
    collapse to the origin when certain anchor kinds relied on the frame
    dimensions.  Passing the actual ``(W, H)`` ensures anchors reflect the scene
    geometry.
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
    data_geo = {"points": pts, "lines": lines, "areas": areas}
    fs = (0.0, 0.0) if frame_size is None else frame_size
    anchors = []
    for lab in labels:
        kind = str(lab.get("anchor_kind", ""))
        idx = lab.get("anchor_index")
        if isinstance(idx, int) and kind:
            qx, qy = anchor_xy(kind, idx, data_geo, fs)
            anchors.append((qx, qy))
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
    force_getter: Optional[Callable[[int], Any]] = None,
    source_getter: Optional[Callable[[int], Dict[str, Any]]] = None,
    metrics_getter: Optional[Callable[[int], Dict[str, Any]]] = None,
    field_getter: Optional[Callable[[int], Any]] = None,
    active_getter: Optional[Callable[[int], Sequence[int]]] = None,
    frames: Optional[Sequence[Dict[str, Any]]] = None,
    field_cmap: str = "viridis",
    boundaries: Optional[Sequence[int]] = None,
    actions: Optional[Sequence[Any]] = None,
    viz: Mapping[str, Any] | None = None,
) -> None:
    """Display an interactive layout viewer.

    Parameters are intentionally kept simple. ``traj`` is expected to be an
    array with shape ``(T, N, 2)`` containing the centre positions of ``N``
    labels over ``T`` iterations. ``rect_wh`` stores the widths and heights of
    the labels. ``force_getter``/``source_getter``/``metrics_getter`` and
    ``field_getter`` are callables returning data for a given iteration.  The
    ``force_getter`` may either return a mapping of force components or a
    tuple containing the force mapping and optionally a scalar field array and
    a source mapping.  Extra elements after the force dictionary are
    interpreted by type: arrays (``numpy`` or ``array_like``) are treated as the
    potential field while ``dict`` instances are stored as raw sources.  This
    allows callers to supply ``(forces, field)``, ``(forces, sources, field)``
    or any other ordering without redundant recomputation.  Missing pieces are
    obtained via ``source_getter``/``field_getter`` when available.
    The presence and type of the field panel are controlled via
    ``viz.panels.field`` and ``viz.field.mode``.  ``field_cmap`` controls the
    colour map used for either representation.

    ``boundaries`` and ``actions`` describe high level actions in the step
    sequence.  When action information is available (either explicit
    ``actions`` or ``boundaries`` describing iteration ranges) an additional
    slider lets the user switch between actions while constraining the iteration
    slider to the corresponding range.
    """

    traj = np.asarray(traj, dtype=float)
    rect_wh = np.asarray(rect_wh, dtype=float)
    T = len(traj)
    N = len(labels)

    viz_config = merge_defaults(viz)
    if viz is None:
        warnings.warn(
            "interactive_view called without viz configuration; using defaults",
            DeprecationWarning,
        )

    panels_cfg = viz_config.get("panels", {})
    show_field = bool(panels_cfg.get("field", True))
    field_cfg = viz_config.get("field", {})
    field_kind = "3d" if field_cfg.get("mode") == "surface3d" else "heatmap"
    field_res = int(field_cfg.get("resolution", 128))
    field_cmap = field_cfg.get("cmap", field_cmap)
    if not show_field:
        field_kind = "none"
        logger.info("[viz] field panel disabled")
    else:
        logger.info(
            "[viz] field panel mode=%s resolution=%d", field_kind, field_res
        )

    SHOW_ORDER = list(ALL_FORCE_KEYS)

    cfg_info = viz_config["info"]
    GLOBAL_FS = cfg_info["row_main_fontsize"] + 1
    LABEL_FS = cfg_info["row_main_fontsize"]
    COMP_FS = cfg_info["row_component_fontsize"]

    THEME = {
        "global_text": "#111111FF",
        "label_text": "#555555FF",
        "force_colors": viz_config["forces"]["colors"],
    }

    def _fmt_sci3(x: Any) -> str:
        try:
            return f"{float(x):.3e}"
        except Exception:
            return "nan"

    def _fmt_force(x: Any) -> str:
        try:
            return f"{float(x):+9.3e}"
        except Exception:
            return "   nan   "

    def _fmt_deg_aligned(x: Any) -> str:
        try:
            return f"{float(x):+6.1f}°"
        except Exception:
            return "   nan°"

    def _fmt_pct_aligned(x: Any) -> str:
        try:
            pct = float(x)
        except Exception:
            return "   nan%"
        s = f"{pct:12.1f}"
        sign = s[0]
        integer = s[1:5].lstrip("0") or "0"
        integer = integer.rjust(4, " ")
        return sign + integer + s[5:] + "%"

    def _compose_info_rows(
        step: int,
        comp_dict: Dict[str, np.ndarray],
        idx: int,
        label_tot: tuple[float, float],
        global_tot: tuple[float, float],
        d_pair: Optional[tuple[float, float]],
        label_name: str,
    ) -> list[tuple]:
        rows: list[tuple] = []
        if metrics_getter is not None:
            try:
                m = metrics_getter(step) or {}
                ui = m.get("ui", {})
                f_style = ui.get("F_style", {})
                g_style = ui.get("g_style", {})
                f_ratio = ui.get("F_ratio")
                g_ratio = ui.get("g_ratio")
                rows.append(
                    (
                        f"|F|_inf = {float(m.get('F_inf', float('nan'))):.3e} / F_tol = {float(m.get('F_tol', float('nan'))):.3e}  (r={float(f_ratio):.2f})",
                        f_style.get("color", THEME["global_text"]),
                        GLOBAL_FS,
                        f_style,
                    )
                )
                rows.append(
                    (
                        f"g_inf   = {float(m.get('g_inf', float('nan'))):.3e} / gtol  = {float(m.get('gtol', float('nan'))):.3e}   (r={float(g_ratio):.2f})",
                        g_style.get("color", THEME["global_text"]),
                        GLOBAL_FS,
                        g_style,
                    )
                )
                rows.append(
                    (
                        f"iter = {m.get('iter', '?')}/{m.get('iter_max', '?')}   solver={m.get('solver', '?')}  stage={m.get('stage', '')}",
                        THEME["global_text"],
                        GLOBAL_FS,
                    )
                )
            except Exception:
                pass

        g_mag, g_ang = global_tot
        rows.append(
            (
                f"{'ALL'.ljust(6)} |F|={_fmt_force(g_mag)}   θ={_fmt_deg_aligned(g_ang)}       ",
                THEME["global_text"],
                GLOBAL_FS,
            )
        )
        if d_pair is not None:
            d_abs, d_rel = d_pair
            rows.append(
                (
                    f"ΔF={_fmt_sci3(d_abs)}   ΔF/F={float(d_rel):.2e}",
                    THEME["global_text"],
                    GLOBAL_FS,
                )
            )

        l_mag, l_ang = label_tot
        name_fmt = label_name[:6].ljust(6)
        rows.append(
            (
                f"{name_fmt} |F|={_fmt_force(l_mag)}   θ={_fmt_deg_aligned(l_ang)}       ",
                THEME["global_text"],
                LABEL_FS,
            )
        )

        tot_mag = l_mag if l_mag > 0 else 1.0
        keys = [k for k in SHOW_ORDER if _as_vec2(comp_dict.get(k)) is not None]
        extras = [k for k, v in comp_dict.items() if _as_vec2(v) is not None and k not in keys]
        keys.extend(extras)
        if keys:
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
                        f"  {name_fmt} |F|={_fmt_force(mag)}   θ={_fmt_deg_aligned(ang)} {_fmt_pct_aligned(pct)}",
                        THEME["force_colors"].get(k, "#555555"),
                        COMP_FS,
                    )
                )
        return rows

    def _draw_info(ax: plt.Axes, rows: list[tuple]) -> None:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        for i, row in enumerate(rows):
            if len(row) == 4:
                text, color, size, style = row

            else:
                text, color, size = row
                style = {}
            delta_height = 0.06 if row[2] == COMP_FS else 0.06
            weight = "bold" if style.get("bold") else "normal"
            ax.text(
                0.01,
                0.99 - i * delta_height,
                text,
                ha="left",
                va="top",
                color=color,
                family="monospace",
                fontsize=size,
                transform=ax.transAxes,
                fontweight=weight,
            )

    # ------------------------------------------------------------------
    # Layout of the figure
    # ------------------------------------------------------------------
    # The overall view is split into a tall main area and a slim bar at the
    # bottom that hosts the iteration slider and step actions.
    fig = plt.figure(figsize=(22, 10))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[28, 1],
        left=0.05, right=0.97, top=0.95, bottom=0.05
    )

    # --- main area -----------------------------------------------------
    if field_kind == "none":
        main = outer[0].subgridspec(1, 2, width_ratios=[5.5, 5])
        ax_layout = fig.add_subplot(main[0, 0])
        centre = main[0, 1].subgridspec(2, 1, height_ratios=[2.5, 2])
        ax_force = fig.add_subplot(centre[0, 0])
        ax_info = fig.add_subplot(centre[1, 0])
        ax_field = None
    else:
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
    bottom = outer[1].subgridspec(1, 2, width_ratios=[2, 3], wspace=0.20)
    slider_ax = fig.add_subplot(bottom[0, 0])
    slider_ax.set_xticks([])
    slider_ax.set_yticks([])
    slider = Slider(slider_ax, "Iter", 0, T - 1, valinit=0, valstep=1)
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
        action_slider = Slider(action_ax, "Act", 0, n_actions - 1, valinit=0, valstep=1)
        action_slider.valtext.set_text("0")
    else:
        action_ax.set_visible(False)

    selected = 0
    patches = []

    _force_cache: Dict[int, Dict[str, np.ndarray]] = {}
    _source_cache: Dict[int, Dict[str, Any]] = {}
    _field_cache: Dict[int, Any] = {}

    def _coerce_force_dict(d: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Return only entries from ``d`` that look like force vectors."""

        if not isinstance(d, dict) or not d:
            return {}
        out: Dict[str, np.ndarray] = {}
        for k, v in d.items():
            arr = _as_vec2(v)
            if arr is not None and len(arr) == N:
                out[k] = arr
        return out

    def _get_forces(step: int) -> Dict[str, np.ndarray]:
        if step in _force_cache:
            return _force_cache[step]
        forces: Dict[str, np.ndarray] = {}
        if callable(force_getter):
            out = force_getter(step)
            if isinstance(out, tuple):
                for extra in out:
                    if isinstance(extra, dict) and not forces:
                        cand = _coerce_force_dict(extra)
                        if cand:
                            forces = cand
                        else:
                            _source_cache[step] = extra
                    elif isinstance(extra, dict):
                        _source_cache[step] = extra
                    else:
                        _field_cache[step] = extra
            elif isinstance(out, dict):
                forces = _coerce_force_dict(out)
        _force_cache[step] = forces
        return forces

    def _get_sources(step: int) -> Dict[str, Any]:
        if step in _source_cache:
            return _source_cache[step]
        if frames is not None and 0 <= step < len(frames):
            sfs = frames[step].get("sources_for_step")
            if sfs is not None:
                _source_cache[step] = sfs
                return sfs
        if callable(source_getter):
            out = source_getter(step)
            if isinstance(out, dict):
                _source_cache[step] = out
                return out
        _get_forces(step)
        return _source_cache.get(step, {})

    def _get_field(step: int) -> Any:
        if step in _field_cache:
            return _field_cache[step]
        if callable(field_getter):
            try:
                _field_cache[step] = field_getter(step, resolution=field_res)
            except TypeError:
                _field_cache[step] = field_getter(step)
            return _field_cache[step]
        _get_forces(step)
        return _field_cache.get(step)

    def _sum_all_forces(forces: Dict[str, np.ndarray]) -> np.ndarray:
        """Sum all force components over all labels and return the net vector.

        Force arrays may contain ``NaN`` values for inactive labels.  Treat
        those entries as zero so that the global sum remains well defined.
        """
        total = None
        for arr in forces.values():
            a = _as_vec2(arr)
            if a is not None:
                a = np.nan_to_num(a, nan=0.0)
                total = a if total is None else total + a
        if total is None:
            return np.zeros(2, dtype=float)
        return np.sum(total, axis=0)

    def _update(step: int) -> None:
        nonlocal patches, selected
        forces_raw = _get_forces(step)
        src_full = _get_sources(step)
        src = dict(src_full) if isinstance(src_full, dict) else {}

        fr = frames[step] if frames and step < len(frames) else None
        ids_solver = (
            fr.get("active_ids_solver") if isinstance(fr, dict) else None
        )
        traj_step = traj[step]
        if ids_solver is not None and traj_step.shape[0] != len(labels):
            if traj_step.shape[0] == len(ids_solver):
                logger.error("[viz] remapping solver subset at step %d", step)
                full = (
                    traj[step - 1]
                    if step > 0 and traj[step - 1].shape[0] == len(labels)
                    else np.zeros((len(labels), 2), float)
                )
                full = np.asarray(full, float).copy()
                for i_local, idx in enumerate(ids_solver):
                    if 0 <= idx < len(labels) and i_local < len(traj_step):
                        full[idx] = traj_step[i_local]
                traj_step = full
            else:
                logger.error(
                    "[viz] active_ids_solver mismatch: %d vs traj rows %d",
                    len(ids_solver),
                    traj_step.shape[0],
                )
                return

        pts_step = src.get("points", points) if src else points
        lns_step = src.get("lines", lines) if src else lines
        ars_step = src.get("areas", areas) if src else areas

        active_ids = list(active_getter(step)) if active_getter else list(range(N))
        labs = [labels[i] for i in active_ids]

        if active_ids != list(range(N)):
            pts_arr = _as_vec2(pts_step)
            lns_arr = None if isinstance(lns_step, (list, tuple)) else _as_vec2(lns_step)
            ars_arr = None if isinstance(ars_step, (list, tuple)) else _as_vec2(ars_step)

            # Gather per-kind anchor indices for active labels
            pts_idx: list[int] = []
            lns_idx: list[int] = []
            ars_idx: list[int] = []
            for lab in labs:
                kind = str(lab.get("anchor_kind", "")).lower()
                idx = lab.get("anchor_index")
                if not isinstance(idx, int):
                    continue
                if kind == "point":
                    pts_idx.append(idx)
                elif kind == "line":
                    lns_idx.append(idx)
                elif kind == "area":
                    ars_idx.append(idx)

            if pts_idx or lns_idx or ars_idx:
                if pts_arr is not None:
                    pts_step = pts_arr[[i for i in pts_idx if 0 <= i < len(pts_arr)]]
                if isinstance(lns_step, (list, tuple)):
                    lns_step = [lns_step[i] for i in lns_idx if 0 <= i < len(lns_step)]
                elif lns_arr is not None:
                    lns_step = lns_arr[[i for i in lns_idx if 0 <= i < len(lns_arr)]]
                if isinstance(ars_step, (list, tuple)):
                    ars_step = [ars_step[i] for i in ars_idx if 0 <= i < len(ars_step)]
                elif ars_arr is not None:
                    ars_step = ars_arr[[i for i in ars_idx if 0 <= i < len(ars_arr)]]
            else:
                if pts_arr is not None:
                    idx = [i for i in active_ids if i < len(pts_arr)]
                    pts_step = pts_arr[idx] if idx else np.zeros((0, 2), float)
                if isinstance(lns_step, (list, tuple)):
                    idx = [i for i in active_ids if i < len(lns_step)]
                    lns_step = [lns_step[i] for i in idx]
                elif lns_arr is not None:
                    idx = [i for i in active_ids if i < len(lns_arr)]
                    lns_step = lns_arr[idx] if idx else np.zeros((0, 2), float)
                if isinstance(ars_step, (list, tuple)):
                    idx = [i for i in active_ids if i < len(ars_step)]
                    ars_step = [ars_step[i] for i in idx]
                elif ars_arr is not None:
                    idx = [i for i in active_ids if i < len(ars_arr)]
                    ars_step = ars_arr[idx] if idx else np.zeros((0, 2), float)

            a_arr = src.get("anchors_xy")
            if a_arr is None:
                a_arr = src.get("anchor_xy")
            a = _as_vec2(a_arr)
            if a is not None:
                idx = [i for i in active_ids if i < len(a)]
                if idx:
                    src["anchors_xy"] = a[idx]
                else:
                    src.pop("anchors_xy", None)
                    src.pop("anchor_xy", None)
            else:
                src.pop("anchors_xy", None)
                src.pop("anchor_xy", None)

        pos_step = traj_step[active_ids]
        wh_step = rect_wh[active_ids]
        if active_ids == list(range(N)):
            forces = forces_raw
        else:
            forces = {k: np.asarray(v, float)[active_ids] for k, v in forces_raw.items()}

        pts_arr = _assert_vec2(pts_step, "points") if pts_step is not None else np.zeros((0, 2), float)
        lns_list = None
        if lns_step is not None:
            iterable = lns_step if isinstance(lns_step, (list, tuple)) else lns_step
            lns_list = [_assert_vec2(pl, "line") for pl in iterable]
        ars_list = None
        if ars_step is not None:
            iterable = ars_step if isinstance(ars_step, (list, tuple)) else ars_step
            ars_list = []
            for poly in iterable:
                if isinstance(poly, dict):
                    poly = poly.get("polygon")
                ars_list.append(_assert_vec2(poly, "area"))

        # [patch-yx-2025-09-05] build full geometry for anchor computation
        pts_full_raw = points if points is not None else (src_full.get("points") if isinstance(src_full, dict) else None)
        pts_full = (
            _assert_vec2(pts_full_raw, "points")
            if pts_full_raw is not None
            else np.zeros((0, 2), float)
        )
        lns_full_raw = lines if lines is not None else (src_full.get("lines") if isinstance(src_full, dict) else None)
        lns_full = None
        if lns_full_raw is not None:
            iterable = lns_full_raw if isinstance(lns_full_raw, (list, tuple)) else lns_full_raw
            lns_full = [_assert_vec2(pl, "line") for pl in iterable]
        ars_full_raw = areas if areas is not None else (src_full.get("areas") if isinstance(src_full, dict) else None)
        ars_full = None
        if ars_full_raw is not None:
            iterable = ars_full_raw if isinstance(ars_full_raw, (list, tuple)) else ars_full_raw
            ars_full = []
            for poly in iterable:
                if isinstance(poly, dict):
                    poly = poly.get("polygon")
                ars_full.append(_assert_vec2(poly, "area"))

        anchors = _compute_anchors(  # [patch-yx-2025-09-05]
            labs,
            points=pts_full,
            lines=lns_full,
            areas=ars_full,
            sources=src_full,
            frame_size=(W, H),
        )

        if selected >= len(active_ids):
            selected = 0

        patches = draw_layout(
            ax_layout,
            pos_step,
            labs,
            wh_step,
            frame_w=W,
            frame_h=H,
            points=pts_arr,
            lines=lns_list,
            areas=ars_list,
            anchors=anchors,
            viz_layout=viz_config["layout"],
        )
        label_id = _label_name(labs[selected], selected)

        terms_to_plot = select_terms_for_arrows(forces, viz_config["forces"])
        label_total = draw_force_panel(
            ax_force,
            forces,
            selected,
            title=label_id,
            terms_to_plot=terms_to_plot,
            viz_forces=viz_config["forces"],
            viz_info=viz_config["info"],
        )

        vec_all = _sum_all_forces(forces)
        g_mag = float(np.hypot(vec_all[0], vec_all[1]))
        g_ang = float(np.degrees(np.arctan2(vec_all[1], vec_all[0])))

        d_pair = None
        if step > 0:
            prev_forces = _get_forces(step - 1)
            prev_vec = _sum_all_forces(prev_forces)
            prev_mag = float(np.hypot(prev_vec[0], prev_vec[1]))
            d_abs = g_mag - prev_mag
            d_rel = d_abs / prev_mag if prev_mag > 0 else 0.0
            d_pair = (d_abs, d_rel)

        if viz_config["forces"].get("info_show_all_terms", True):
            comps_info = normalize_comps_for_info(forces, len(active_ids))
        else:
            comps_info = forces
        rows = _compose_info_rows(
            step, comps_info, selected, label_total, (g_mag, g_ang), d_pair, label_id
        )
        _draw_info(ax_info, rows)

        pts_arr = _as_vec2(pts_step)
        lns_arr = _as_vec2(lns_step) if not isinstance(lns_step, (list, tuple)) else None
        ars_arr = _as_vec2(ars_step) if not isinstance(ars_step, (list, tuple)) else None
        logger.info(
            "[viz] step=%d geom pts/lns/ars=%d/%d/%d | arrows terms=%d | info terms=%d",
            step,
            0 if pts_arr is None else len(pts_arr),
            len(lns_step) if isinstance(lns_step, (list, tuple)) else (0 if lns_arr is None else len(lns_arr)),
            len(ars_step) if isinstance(ars_step, (list, tuple)) else (0 if ars_arr is None else len(ars_arr)),
            len(terms_to_plot),
            len(comps_info.keys()),
        )

        if ax_field is not None:
            field = _get_field(step)
            draw_field_panel(ax_field, field, W, H, field_kind, field_cmap)

        fig.canvas.draw_idle()

    def _on_pick(event):
        nonlocal selected
        for i, patch in patches:
            if event.artist is patch:
                selected = i
                _update(int(slider.val))
                break

    def _on_slider(val):
        step = int(max(0, min(val, T - 1)))
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
        hi = min(hi, T - 1)
        lo = max(0, min(lo, hi))
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

