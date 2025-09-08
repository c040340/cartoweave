from __future__ import annotations

# 1) Choose backend BEFORE importing pyplot
from .backend import setup_matplotlib_backend

_backend = setup_matplotlib_backend()

import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from matplotlib.widgets import Slider
from matplotlib.artist import Artist
from types import SimpleNamespace

from .defaults import VIZ_DEFAULTS
from cartoweave.config.loader import load_viz_defaults
from . import panels
from .layout_style import get_layout_style_from_cfg


def _merge_viz_cfg(user_cfg: dict | None) -> dict:
    cfg = VIZ_DEFAULTS.copy()
    cfg.update(load_viz_defaults() or {})
    cfg.update(user_cfg or {})
    '''cfg = dict(VIZ_DEFAULTS) if isinstance(VIZ_DEFAULTS, dict) else {}
    if user_cfg:
        panels_default = cfg.get("panels", {})
        panels_user = user_cfg.get("panels", {})
        cfg.update(user_cfg)
        if panels_default or panels_user:
            m = dict(panels_default)
            m.update(panels_user)
            cfg["panels"] = m'''
    return cfg

@dataclass
class ViewAxes:
    fig: plt.Figure
    ax_layout: plt.Axes
    ax_force: plt.Axes
    ax_info: plt.Axes
    ax_field: Optional[plt.Axes]
    ax_bar_action: plt.Axes
    ax_bar_iter: plt.Axes


def build_view_axes(field_kind: str = "3d") -> ViewAxes:
    """Create figure and axes with fixed layout ratios.

    Parameters
    ----------
    field_kind: str
        Kind of field panel. Use "none" to disable the field panel.
    """

    fig = plt.figure(figsize=(22, 10))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[28, 1],
        left=0.05,
        right=0.97,
        top=0.95,
        bottom=0.05,
    )

    main = outer[0]
    bar = outer[1].subgridspec(1, 2, width_ratios=[1, 3])
    bar_action = bar[0]
    bar_iter = bar[1]

    if field_kind == "none":
        gs_main = main.subgridspec(1, 2, width_ratios=[5.5, 5])
        ax_layout = fig.add_subplot(gs_main[0, 0])

        centre = gs_main[0, 1].subgridspec(2, 1, height_ratios=[2.5, 2])
        ax_force = fig.add_subplot(centre[0, 0])
        ax_info = fig.add_subplot(centre[1, 0])
        ax_field = None
    else:
        gs_main = main.subgridspec(1, 3, width_ratios=[5.5, 5, 7.5])
        ax_layout = fig.add_subplot(gs_main[0, 0])

        centre = gs_main[0, 1].subgridspec(2, 1, height_ratios=[2.5, 2])
        ax_force = fig.add_subplot(centre[0, 0])
        ax_info = fig.add_subplot(centre[1, 0])
        projection = "3d" if field_kind == "3d" else None
        ax_field = fig.add_subplot(gs_main[0, 2], projection=projection)

    ax_bar_action = fig.add_subplot(bar_action)
    ax_bar_iter = fig.add_subplot(bar_iter)

    return ViewAxes(
        fig=fig,
        ax_layout=ax_layout,
        ax_force=ax_force,
        ax_info=ax_info,
        ax_field=ax_field,
        ax_bar_action=ax_bar_action,
        ax_bar_iter=ax_bar_iter,
    )


def show_vp(view_pack, viz_cfg: dict | None = None):
    """Multi-panel ViewPack viewer."""

    cfg = _merge_viz_cfg(viz_cfg or {})
    style = get_layout_style_from_cfg(cfg)
    frames = getattr(view_pack, "frames", []) or []
    T = len(frames)
    if T == 0:
        raise ValueError("ViewPack.frames is empty")

    passes = getattr(view_pack, "passes", []) or []
    if not passes:
        passes = [SimpleNamespace(pass_id=0, pass_name="", t_start=0, t_end=T)]

    pass_of_t: List[int] = [0] * T
    for idx, p in enumerate(passes):
        for tt in range(p.t_start, p.t_end):
            if 0 <= tt < T:
                pass_of_t[tt] = idx

    panels_cfg = cfg.get("panels", {})
    field_kind = "3d" if panels_cfg.get("field", False) else "none"
    axes = build_view_axes(field_kind=field_kind)
    fig = axes.fig
    ax_layout = axes.ax_layout
    ax_force = axes.ax_force
    ax_info = axes.ax_info
    ax_field = axes.ax_field
    ax_bar_action = axes.ax_bar_action
    ax_bar_iter = axes.ax_bar_iter

    if not panels_cfg.get("forces", True):
        ax_force.set_visible(False)
    if not panels_cfg.get("info", True):
        ax_info.set_visible(False)

    state: Dict[str, int | None] = {"t": 0, "sel": None, "pass": 0}
    patches: List[Tuple[int, Artist]] = []

    def _format_meta(meta: Dict[str, Any] | None) -> str:
        if not isinstance(meta, dict):
            return ""
        line1 = f"{meta.get('pass_name', '')}:{meta.get('frame_in_pass', '')}"
        events = meta.get("events") or []
        line2 = ""
        if isinstance(events, list) and events:
            ev = events[-1]
            if isinstance(ev, dict):
                if ev.get("kind") == "optimizer_step":
                    algo = ev.get("algo", "")
                    it = ev.get("iter_in_algo", "")
                    line2 = f"{algo} iter {it}"
                else:
                    p = ev.get("pass") or ev.get("kind") or ""
                    info = ev.get("info") or ""
                    line2 = f"{p} {info}".strip()
        return f"{line1}\n{line2}".rstrip()

    def redraw_layout(t: int):
        ax = ax_layout
        ax.clear()
        if hasattr(panels, "draw_layout"):
            patches[:] = panels.draw_layout(ax, view_pack, t, style=style) or []
        sel = state.get("sel")
        if sel is not None:
            for i, art in patches:
                if i == sel:
                    art.set_edgecolor("#ff0000")
                    art.set_linewidth(style.label_edge_width * 2)
                    break
        ax.figure.canvas.draw_idle()

    def redraw_forces(t: int):
        if not ax_force.get_visible():
            return
        ax = ax_force
        ax.clear()
        if hasattr(panels, "draw_forces"):
            panels.draw_forces(ax, view_pack, t, cfg, state.get("sel"))
        else:
            ax.text(
                0.5,
                0.5,
                "forces (placeholder)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("forces")
        ax.figure.canvas.draw_idle()

    def redraw_info(t: int):
        if not ax_info.get_visible():
            return
        ax = ax_info
        ax.clear()
        if hasattr(panels, "draw_info"):
            panels.draw_info(ax, view_pack, t, cfg, state.get("sel"))
        else:
            fr = frames[t]
            comps = getattr(fr, "comps", None)
            ax.set_axis_off()
            if isinstance(comps, dict):
                y = 0.95
                for k, v in sorted(comps.items()):
                    ax.text(
                        0.02,
                        y,
                        f"{k}: {v:.6g}",
                        transform=ax.transAxes,
                        va="top",
                        family="monospace",
                    )
                    y -= 0.06
            ax.set_title("info", pad=8)
        ax.figure.canvas.draw_idle()

    def redraw_field(t: int):
        if ax_field is None or not ax_field.get_visible():
            return
        ax = ax_field
        ax.clear()
        if hasattr(panels, "draw_field"):
            panels.draw_field(ax, view_pack, t, cfg)
        else:
            ax.text(
                0.5,
                0.5,
                "field (placeholder)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("field")
        ax.figure.canvas.draw_idle()

    def set_t(t_new: int, sync_slider: bool = True):
        t_new = max(0, min(T - 1, int(round(t_new))))
        if t_new == state["t"]:
            return
        state["t"] = t_new
        p_idx = pass_of_t[t_new]
        state["pass"] = p_idx
        redraw_layout(t_new)
        redraw_forces(t_new)
        redraw_info(t_new)
        redraw_field(t_new)
        fr = frames[t_new]
        meta_text.set_text(_format_meta(getattr(fr, "meta", None)))
        if sync_slider:
            try:
                slider_action.set_val(p_idx)
                p = passes[p_idx]
                local_idx = t_new - p.t_start
                slider_iter.valmax = max(p.t_end - p.t_start - 1, 0)
                slider_iter.ax.set_xlim(0, slider_iter.valmax)
                slider_iter.set_val(local_idx)
            except Exception:
                pass

    ax_bar_action.clear()
    ax_bar_iter.clear()
    slider_action = Slider(ax_bar_action, "pass", 0, len(passes) - 1, valinit=0, valstep=1)
    first_len = max(passes[0].t_end - passes[0].t_start - 1, 0)
    slider_iter = Slider(ax_bar_iter, "frame", 0, first_len, valinit=0, valstep=1)
    meta_text = ax_bar_iter.text(
        1.0,
        1.0,
        "",
        transform=ax_bar_iter.transAxes,
        ha="right",
        va="bottom",
    )

    def on_action(val):
        p_idx = int(val)
        p = passes[p_idx]
        slider_iter.valmax = max(p.t_end - p.t_start - 1, 0)
        slider_iter.ax.set_xlim(0, slider_iter.valmax)
        slider_iter.set_val(0)
        set_t(p.t_start, sync_slider=False)

    def on_iter(val):
        p_idx = int(slider_action.val)
        p = passes[p_idx]
        t = p.t_start + int(val)
        set_t(t, sync_slider=False)

    slider_action.on_changed(on_action)
    slider_iter.on_changed(on_iter)

    def on_key(event):
        if event.key in ("left", "a"):
            set_t(state["t"] - 1)
        elif event.key in ("right", "d"):
            set_t(state["t"] + 1)
        elif event.key == "home":
            set_t(0)
        elif event.key == "end":
            set_t(T - 1)

    def on_pick(event):
        artist = event.artist
        for i, art in patches:
            if artist is art:
                state["sel"] = i
                redraw_layout(state["t"])
                redraw_info(state["t"])
                break

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("pick_event", on_pick)

    set_t(0, sync_slider=True)
    plt.show()


__all__ = ["show_vp", "build_view_axes", "ViewAxes"]

