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
import os

import matplotlib

# Use an interactive backend when a display is available.  This keeps the
# example functional in headless environments (e.g. automated tests) where
# ``TkAgg`` cannot be loaded.
cwd = os.path.abspath(os.getcwd())
if os.environ.get("DISPLAY") or "yuanzheng" in cwd:
    try:  # pragma: no cover - depends on environment
        matplotlib.use("TkAgg")
    except Exception:  # pragma: no cover - backend unavailable
        pass
else:
    matplotlib.use("Agg")

from cartoweave.api import solve_frame
from cartoweave.config.presets import default_cfg, area_pack_cfg
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

cfg_base = merge(default_cfg(), area_pack_cfg(), viz(show=True, field_kind="heatmap"))
schedule = [
    {"name": "warmup_no_anchor", "scale": {"anchor.k.spring": 0.0}},
    {"name": "main_solve"},
]


# ---------------------------------------------------------------------------
# Solve the schedule for each timeline action while recording positions
# ---------------------------------------------------------------------------

id_to_idx = {lab["id"]: i for i, lab in enumerate(scene["labels"])}

P_all = np.full_like(scene["labels_init"], np.nan)
traj = []
sub_scenes: list[dict] = []
active_idx_per_step: list[list[int]] = []
sources_per_step: list[dict] = []
energies_per_step: list[float] = []
boundaries = [0]
active_ids: set[str] = set()

for act in actions:
    op = act.get("op")
    eid = str(act.get("id"))
    if op == "appear":
        active_ids.add(eid)
        idx = id_to_idx.get(eid)
        if idx is not None:
            P_all[idx] = scene["labels_init"][idx]
    elif op == "hide":
        active_ids.discard(eid)
        idx = id_to_idx.get(eid)
        if idx is not None:
            P_all[idx] = np.nan
    elif op == "change":
        pass  # mode changes ignored in this simple example

    active_idx = sorted(id_to_idx[iid] for iid in active_ids)
    pt_idx = [int(i[1:]) for i in active_ids if i.startswith("p")]
    ln_idx = [int(i[1:]) for i in active_ids if i.startswith("l")]
    ar_idx = [int(i[1:]) for i in active_ids if i.startswith("a")]

    sub_scene = dict(
        frame=scene["frame"],
        frame_size=scene["frame_size"],
        points=scene["points"][pt_idx],
        lines=scene["lines"][ln_idx],
        areas=[scene["areas"][i] for i in ar_idx],
        labels_init=P_all[active_idx],
        WH=scene["WH"][active_idx],
        anchors=scene["anchors"][active_idx],
        labels=[scene["labels"][i] for i in active_idx],
    )

    for step in schedule:
        cfg_step = dict(cfg_base)
        for k, s in step.get("scale", {}).items():
            if k in cfg_step:
                cfg_step[k] = float(cfg_step[k]) * float(s)
        for k, v in step.get("override", {}).items():
            cfg_step[k] = v

        P_opt, info = solve_frame(sub_scene, cfg_step, mode="hybrid")
        hist = info.get("history", {}).get("positions", [])
        hist_E = info.get("history", {}).get("energies", [])
        for i, P_hist in enumerate(hist):
            if traj and i == 0:
                continue  # skip duplicate starting point
            P_all[active_idx] = P_hist
            anchors_full = np.full_like(P_all, np.nan)
            anchors_full[active_idx] = sub_scene["anchors"]
            sources_per_step.append(
                {
                    "points": sub_scene["points"],
                    "lines": [seg.reshape(2, 2) for seg in sub_scene["lines"]],
                    "areas": [a["polygon"] for a in sub_scene["areas"]],
                    "anchors_xy": anchors_full,
                }
            )
            traj.append(P_all.copy())
            sub_scenes.append(dict(sub_scene))
            active_idx_per_step.append(active_idx)
            E_hist = hist_E[i] if i < len(hist_E) else float("nan")
            energies_per_step.append(float(E_hist))

        P_all[active_idx] = P_opt
        sub_scene["labels_init"] = P_opt

    boundaries.append(len(traj))

traj = np.stack(traj, axis=0)

print("Ran actions:", [s["name"] for s in schedule])
print("Action boundaries:", boundaries)
print("Final positions:\n", traj[-1])


# ---------------------------------------------------------------------------
# Optional interactive visualisation
# ---------------------------------------------------------------------------

if cfg_base.get("viz.show", False):
    metrics_cache: dict[int, dict[str, float]] = {}

    def force_getter(step: int):
        sc = sub_scenes[step]
        act_idx = active_idx_per_step[step]
        P_step = traj[step][act_idx]
        holder: dict[str, dict[str, np.ndarray]] = {}

        def rec(P, E, comps, sources):
            holder["comps"] = comps

        E, G, _ = energy_and_grad_fullP(sc, P_step, cfg_base, record=rec)
        metrics_cache[step] = {
            "f": float(E),
            "gnorm_inf": float(np.abs(G).max()) if G.size else float("nan"),
        }
        comps = holder.get("comps", {})
        expanded: dict[str, np.ndarray] = {}
        for name, arr in comps.items():
            arr_full = np.full_like(traj[step], np.nan)
            arr_full[act_idx] = arr
            expanded[name] = arr_full
        return expanded

    def source_getter(step: int):
        return sources_per_step[step]

    def metrics_getter(step: int):
        E = energies_per_step[step]
        prev = energies_per_step[step - 1] if step > 0 else E
        dE = E - prev
        rel = dE / prev if step > 0 and prev != 0 else float("nan")
        m = metrics_cache.get(step, {})
        return {
            "iter": step,
            "f": E,
            "df": dE,
            "rel_df": rel,
            "gnorm_inf": m.get("gnorm_inf", float("nan")),
        }

    interactive_view(
        traj,
        scene["labels"],
        scene["WH"],
        scene["points"],
        scene["lines"],
        [a["polygon"] for a in scene["areas"]],
        W=scene["frame_size"][0],
        H=scene["frame_size"][1],
        force_getter=force_getter,
        source_getter=source_getter,
        metrics_getter=metrics_getter,
        field_kind=cfg_base.get("viz.field.kind", "heatmap"),
        field_cmap=cfg_base.get("viz.field.cmap", "viridis"),
        actions=actions,
        boundaries=boundaries,
    )

