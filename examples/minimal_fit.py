"""Minimal end-to-end example with optional visualization.

This script solves a tiny scene with three labels – each anchored to a point,
line and area respectively – and launches the interactive viewer if
``viz.show`` is enabled in the config.  The viewer computes forces on demand
to keep the example lightweight.
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from cartoweave.api import solve_frame
from cartoweave.config.presets import default_cfg
from cartoweave.config.utils import merge, viz
from cartoweave.engine.core_eval import energy_and_grad_fullP, scalar_potential_field
from cartoweave.viz import interactive_view

from cartoweave.utils.logging import configure_logging
configure_logging(True, 4)

# ---------------------------------------------------------------------------
# Scene setup: labels anchored to a point, a line and a triangular area
# ---------------------------------------------------------------------------

scene = dict(
    frame=0,
    frame_size=(1080, 1920),
    # single point
    points=np.array([[100.0, 100.0]], float),
    # single line segment represented as (x0, y0, x1, y1)
    lines=np.array([[200.0, 80.0, 300.0, 80.0]], float),
    # single triangular area
    areas=[{"polygon": np.array([[400.0, 60.0], [450.0, 140.0], [350.0, 140.0]], float)}],
    # initial label positions
    labels_init=np.array([
        [120.0, 100.0],  # near the point
        [260.0, 60.0],   # above the line
        [400.0, 100.0],  # inside the triangle
    ], float),
    WH=np.array([[60.0, 24.0], [60.0, 24.0], [60.0, 24.0]], float),
    labels=[
        {"anchor_kind": "point", "anchor_index": 0},
        {"anchor_kind": "line", "anchor_index": 0},
        {"anchor_kind": "area", "anchor_index": 0},
    ],
)


# ---------------------------------------------------------------------------
# Configuration: merge defaults with visualization options
# ---------------------------------------------------------------------------

cfg = merge(default_cfg(), viz(show=True, field_kind="3d"))


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

P_opt, info = solve_frame(scene, cfg, mode="semi_newton")
print("P_opt:", P_opt)


# ---------------------------------------------------------------------------
# Optional interactive visualization
# ---------------------------------------------------------------------------

if cfg.get("viz.show", False):
    traj = np.asarray(info.get("history", {}).get("positions", [P_opt]))
    lines_draw = [seg.reshape(2, 2) for seg in scene["lines"]]
    areas_draw = [a["polygon"] for a in scene["areas"]]

    def compute_force(step: int):
        holder: dict[str, dict[str, np.ndarray]] = {}

        def rec(P, E, comps, sources):
            holder["comps"] = comps
            holder["sources"] = sources

        energy_and_grad_fullP(scene, traj[step], cfg, record=rec)

        field = scalar_potential_field(scene, traj[step], cfg)
        # return as (forces, field, sources)
        return holder.get("comps", {}), field, holder.get("sources", {})

    interactive_view(
        traj,
        scene["labels"],
        scene["WH"],
        scene["points"],
        lines_draw,
        areas_draw,
        W=scene["frame_size"][0],
        H=scene["frame_size"][1],
        force_getter=compute_force,
        field_kind=cfg.get("viz.field.kind", "3d"),
        field_cmap=cfg.get("viz.field.cmap", "viridis"),
    )
