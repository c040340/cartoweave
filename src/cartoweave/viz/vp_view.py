from __future__ import annotations

# 1) Choose backend BEFORE importing pyplot
from .backend import setup_matplotlib_backend

_backend = setup_matplotlib_backend()

import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from matplotlib.widgets import Slider

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
    ax_bar: plt.Axes


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
    bar = outer[1]

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

    ax_bar = fig.add_subplot(bar)

    return ViewAxes(
        fig=fig,
        ax_layout=ax_layout,
        ax_force=ax_force,
        ax_info=ax_info,
        ax_field=ax_field,
        ax_bar=ax_bar,
    )


def show_vp(view_pack, viz_cfg: dict | None = None):
    """Multi-panel ViewPack viewer."""

    cfg = _merge_viz_cfg(viz_cfg or {})
    style = get_layout_style_from_cfg(cfg)
    frames = getattr(view_pack, "frames", []) or []
    T = len(frames)
    if T == 0:
        raise ValueError("ViewPack.frames is empty")

    panels_cfg = cfg.get("panels", {})
    field_kind = "3d" if panels_cfg.get("field", False) else "none"
    axes = build_view_axes(field_kind=field_kind)
    fig = axes.fig
    ax_layout = axes.ax_layout
    ax_force = axes.ax_force
    ax_info = axes.ax_info
    ax_field = axes.ax_field
    ax_bar = axes.ax_bar

    if not panels_cfg.get("forces", True):
        ax_force.set_visible(False)
    if not panels_cfg.get("info", True):
        ax_info.set_visible(False)

    state = {"t": 0}

    def redraw_layout(t: int):
        ax = ax_layout
        ax.clear()
        if hasattr(panels, "draw_layout"):
            panels.draw_layout(ax, view_pack, t, style=style)
        ax.figure.canvas.draw_idle()

    def redraw_forces(t: int):
        if not ax_force.get_visible():
            return
        ax = ax_force
        ax.clear()
        if hasattr(panels, "draw_forces"):
            panels.draw_forces(ax, view_pack, t, cfg)
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
            panels.draw_info(ax, view_pack, t, cfg)
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
        redraw_layout(t_new)
        redraw_forces(t_new)
        redraw_info(t_new)
        redraw_field(t_new)
        if sync_slider:
            try:
                slider.set_val(t_new)
            except Exception:
                pass

    ax_bar.clear()
    slider = Slider(ax_bar, "frame", 0, T - 1, valinit=0, valstep=1)

    try:
        last_ai = None
        for i, fr in enumerate(frames):
            ai = None
            meta = getattr(fr, "meta", None)
            if isinstance(meta, dict):
                ai = meta.get("action_index")
            if ai is not None and ai != last_ai:
                ax_bar.axvline(i, color="0.7", lw=0.8, alpha=0.8)
                last_ai = ai
    except Exception:
        pass

    def on_slide(val):
        set_t(int(val), sync_slider=False)

    slider.on_changed(on_slide)

    def on_key(event):
        if event.key in ("left", "a"):
            set_t(state["t"] - 1)
        elif event.key in ("right", "d"):
            set_t(state["t"] + 1)
        elif event.key == "home":
            set_t(0)
        elif event.key == "end":
            set_t(T - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    set_t(0, sync_slider=True)
    plt.show()


__all__ = ["show_vp", "build_view_axes", "ViewAxes"]

