"""Minimal end-to-end example with optional visualization.

This script solves a tiny scene with one label and launches the interactive
viewer if ``viz.show`` is enabled in the config.  The viewer uses forces
computed on demand to keep the example lightweight.
"""

from __future__ import annotations

import numpy as np

from cartoweave.api import solve_frame
from cartoweave.config.presets import default_cfg
from cartoweave.config.utils import merge, viz
from cartoweave.engine.core_eval import energy_and_grad_fullP
from cartoweave.viz import interactive_view


# ---------------------------------------------------------------------------
# Scene setup: one label anchored to a point
# ---------------------------------------------------------------------------

scene = dict(
    frame=0,
    frame_size=(800, 600),
    points=np.array([[100.0, 100.0]], float),
    lines=np.zeros((0, 4), float),
    areas=[],
    labels_init=np.array([[120.0, 100.0]], float),
    WH=np.array([[60.0, 24.0]], float),
    labels=[{"anchor_kind": "point", "anchor_index": 0}],
)


# ---------------------------------------------------------------------------
# Configuration: merge defaults with visualization options
# ---------------------------------------------------------------------------

cfg = merge(default_cfg(), viz(show=True, field_kind="heatmap"))


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

    def compute_force(step: int):
        holder: dict[str, dict[str, np.ndarray]] = {}

        def rec(P, E, comps, sources):
            holder["comps"] = comps
            holder["sources"] = sources

        energy_and_grad_fullP(scene, traj[step], cfg, record=rec)
        return holder.get("comps", {}), holder.get("sources", {})

    interactive_view(
        traj,
        scene["labels"],
        scene["WH"],
        scene["points"],
        scene["lines"],
        scene["areas"],
        W=scene["frame_size"][0],
        H=scene["frame_size"][1],
        force_getter=compute_force,
        field_kind=cfg.get("viz.field.kind", "3d"),
        field_cmap=cfg.get("viz.field.cmap", "viridis"),
    )

