"""Generate or reuse a random scene, then run a two-step optimisation.

This example demonstrates the full pipeline using the random data utilities.
The counts of points/lines/areas as well as whether to regenerate the scene or
reuse the last cached one are exposed as variables so users can tweak them.
A random appear/change/hide timeline is produced via :func:`make_timeline` and
cached together with the scene to guarantee reproducibility when ``use_random``
is ``False``.
"""

from __future__ import annotations

import numpy as np
import time

import matplotlib
matplotlib.use("TkAgg")

from cartoweave.api import solve_frame
from cartoweave.config.presets import default_cfg
from cartoweave.config.utils import merge, viz
from cartoweave.data.random import get_scene, make_timeline
from cartoweave.engine.core_eval import energy_and_grad_fullP
from cartoweave.viz import interactive_view


# ---------------------------------------------------------------------------
# Random scene configuration
# ---------------------------------------------------------------------------

# Users can tweak these parameters.
N_POINTS = 5
N_LINES = 2
N_AREAS = 1
SEED = int(time.time_ns() // (2**32))

# Whether to generate a new random scene or reuse the last cached one.
# When ``use_random`` is ``False``, point/line/area geometry as well as labels
# and timeline will be loaded from the cache generated in the previous run.
USE_RANDOM = False


def build_timeline(data: dict) -> list[dict]:
    """Wrapper around :func:`make_timeline` for :func:`get_scene`.

    ``get_scene`` passes the full scene dict.  We only need element IDs here.
    """

    Np = data["points"].shape[0]
    Nl = data["lines"].shape[0]
    Na = len(data["areas"])
    pt_ids = [{"id": f"p{i}"} for i in range(Np)]
    li_ids = [{"id": f"l{i}"} for i in range(Nl)]
    ar_ids = [{"id": f"a{i}"} for i in range(Na)]
    return make_timeline(pt_ids, li_ids, ar_ids, base_modes={}, seed=SEED)


scene = get_scene(
    use_random=USE_RANDOM,
    with_timeline=True,
    auto_make_timeline_if_missing=True,
    make_timeline_fn=build_timeline,
    seed=SEED,
    canvas_size=(1080.0, 1920.0),
    n_points=N_POINTS,
    n_lines=N_LINES,
    n_areas=N_AREAS,
)

actions = scene.get("timeline", [])
print("Timeline actions:")
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

