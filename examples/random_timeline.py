"""Generate a random scene, run a two-step optimisation and visualise.

This example demonstrates the full pipeline using the random data utilities.
The numbers of points, lines and areas are exposed as variables so users can
tweak them freely.  A random appear/change/hide timeline is also produced for
reference.
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from cartoweave.api import solve_frame
from cartoweave.config.presets import default_cfg
from cartoweave.config.utils import merge, viz
from cartoweave.data.random import generate_scene, make_timeline
from cartoweave.engine.core_eval import energy_and_grad_fullP
from cartoweave.viz import interactive_view


# ---------------------------------------------------------------------------
# Random scene configuration
# ---------------------------------------------------------------------------

# Users can edit these numbers to control the amount of random geometry.
N_POINTS = 5
N_LINES = 2
N_AREAS = 1
SEED = 0

scene = generate_scene(
    canvas_size=(800.0, 600.0),
    n_points=N_POINTS,
    n_lines=N_LINES,
    n_areas=N_AREAS,
    seed=SEED,
)

# Also create a random action timeline (appear/change/hide) for reference.
point_ids = [{"id": f"p{i}"} for i in range(N_POINTS)]
line_ids = [{"id": f"l{i}"} for i in range(N_LINES)]
area_ids = [{"id": f"a{i}"} for i in range(N_AREAS)]
actions = make_timeline(point_ids, line_ids, area_ids, base_modes={}, seed=SEED)
print("Generated timeline actions:")
for act in actions:
    print("  ", act)


# ---------------------------------------------------------------------------
# Configuration and a timeline with at least two optimisation steps
# ---------------------------------------------------------------------------

cfg_base = merge(default_cfg(), viz(show=True, field_kind="heatmap"))
schedule = [
    {"name": "warmup_no_anchor", "scale": {"anchor.k.spring": 0.0}},
    {"name": "main_solve"},
]


# ---------------------------------------------------------------------------
# Solve the schedule while recording positions
# ---------------------------------------------------------------------------

traj = []
P_init = scene["labels_init"]
for step in schedule:
    cfg_step = dict(cfg_base)
    for k, s in step.get("scale", {}).items():
        if k in cfg_step:
            cfg_step[k] = float(cfg_step[k]) * float(s)
    for k, v in step.get("override", {}).items():
        cfg_step[k] = v

    scene["labels_init"] = P_init
    P_opt, _ = solve_frame(scene, cfg_step, mode="hybrid")
    traj.append(P_opt.copy())
    P_init = P_opt

traj = np.stack(traj, axis=0)

print("Ran actions:", [s["name"] for s in schedule])
print("Final positions:\n", traj[-1])


# ---------------------------------------------------------------------------
# Optional interactive visualisation
# ---------------------------------------------------------------------------

if cfg_base.get("viz.show", False):
    def compute_force(step: int):
        holder: dict[str, dict[str, np.ndarray]] = {}

        def rec(P, E, comps, sources):
            holder["comps"] = comps
            holder["sources"] = sources

        energy_and_grad_fullP(scene, traj[step], cfg_base, record=rec)
        return holder.get("comps", {}), holder.get("sources", {})

    lines_draw = [seg.reshape(2, 2) for seg in scene["lines"]]
    areas_draw = [a["polygon"] for a in scene["areas"]]

    interactive_view(
        traj,
        scene["labels"],
        scene["WH"],
        scene["points"],
        lines_draw,
        areas_draw,
        W=scene["frame_size"][0],
        H=scene["frame_size"][1],
        force_getter=lambda step: compute_force(step)[0],
        source_getter=lambda step: compute_force(step)[1],
        field_kind=cfg_base.get("viz.field.kind", "heatmap"),
        field_cmap=cfg_base.get("viz.field.cmap", "viridis"),
    )


