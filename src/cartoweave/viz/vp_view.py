from __future__ import annotations

# 1) Choose backend BEFORE importing pyplot
from .backend import setup_matplotlib_backend

_backend = setup_matplotlib_backend()

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.widgets import Slider

from .defaults import VIZ_DEFAULTS
from . import panels


def _merge_viz_cfg(user_cfg: dict | None) -> dict:
    cfg = dict(VIZ_DEFAULTS) if isinstance(VIZ_DEFAULTS, dict) else {}
    if user_cfg:
        panels_default = cfg.get("panels", {})
        panels_user = user_cfg.get("panels", {})
        cfg.update(user_cfg)
        if panels_default or panels_user:
            m = dict(panels_default)
            m.update(panels_user)
            cfg["panels"] = m
    return cfg


def _build_axes(fig, T: int, cfg: dict):
    gs = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        height_ratios=[24, 1],
        width_ratios=[1, 1, 1],
        hspace=0.25,
        wspace=0.25,
    )

    ax_layout = fig.add_subplot(gs[0, 0])

    gs_mid = GridSpecFromSubplotSpec(
        nrows=2, ncols=1, subplot_spec=gs[0, 1], height_ratios=[1, 1], hspace=0.25
    )
    ax_forces = fig.add_subplot(gs_mid[0, 0])
    ax_info = fig.add_subplot(gs_mid[1, 0])

    ax_field = fig.add_subplot(gs[0, 2])
    ax_slider = fig.add_subplot(gs[1, :])

    panels_cfg = cfg.get("panels", {})
    if not panels_cfg.get("forces", True):
        ax_forces.set_visible(False)
    if not panels_cfg.get("info", True):
        ax_info.set_visible(False)
    if not panels_cfg.get("field", True):
        ax_field.set_visible(False)

    return {
        "layout": ax_layout,
        "forces": ax_forces,
        "info": ax_info,
        "field": ax_field,
        "slider": ax_slider,
    }


def show_vp(view_pack, viz_cfg: dict | None = None):
    """Multi-panel ViewPack viewer."""

    cfg = _merge_viz_cfg(viz_cfg or {})
    frames = getattr(view_pack, "frames", []) or []
    T = len(frames)
    if T == 0:
        raise ValueError("ViewPack.frames is empty")

    fig = plt.figure(figsize=(12, 8))
    axes = _build_axes(fig, T, cfg)
    ax_layout = axes["layout"]
    ax_forces = axes["forces"]
    ax_info = axes["info"]
    ax_field = axes["field"]
    ax_slider = axes["slider"]

    state = {"t": 0}

    def redraw_layout(t: int):
        ax = ax_layout
        ax.clear()
        if hasattr(panels, "draw_layout"):
            panels.draw_layout(ax, view_pack, t, cfg)
        ax.figure.canvas.draw_idle()

    def redraw_forces(t: int):
        if not ax_forces.get_visible():
            return
        ax = ax_forces
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
        if not ax_field.get_visible():
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

    ax_slider.clear()
    slider = Slider(ax_slider, "frame", 0, T - 1, valinit=0, valstep=1)

    try:
        last_ai = None
        for i, fr in enumerate(frames):
            ai = None
            meta = getattr(fr, "meta", None)
            if isinstance(meta, dict):
                ai = meta.get("action_index")
            if ai is not None and ai != last_ai:
                ax_slider.axvline(i, color="0.7", lw=0.8, alpha=0.8)
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


__all__ = ["show_vp"]

