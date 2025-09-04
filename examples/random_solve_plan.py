"""Random scene + short solve plan using scene script.

- Generates or loads a cached random scene and its scene script.
- Loads layered config with calibration profiles turned off for determinism.
- Executes a two-stage solve plan over all script steps.
"""

from __future__ import annotations
from typing import Dict, Any
import os
import numpy as np

from cartoweave.viz.backend import use_compatible_backend

use_compatible_backend()

from cartoweave.config.layering import (
    load_base_cfg,
    apply_calib_profile,
    apply_shape_profile,
)
from cartoweave.data.random import get_scene
from cartoweave.api import solve_scene_script

try:  # optional viewer
    from cartoweave.viz.view import interactive_view
except Exception:  # pragma: no cover - viewer not installed
    interactive_view = None

CACHE_PATH = os.environ.get("CARTOWEAVE_EXAMPLE_CACHE", "examples/_scene_cache.npz")
GENERATE_NEW = bool(int(os.environ.get("CARTOWEAVE_GENERATE_NEW", "0")))


def _build_cfg() -> Dict[str, Any]:
    cfg = load_base_cfg()
    apply_calib_profile(cfg, cfg.get("calib.k.profile", "default"), fill_only=True)

    # Apply one shape profile so base forces exist but keep gates OFF for
    # deterministic lightweight runs.
    apply_shape_profile(cfg, cfg.get("calib.shape.name", "default"), enable=True)
    cfg["calib.shape.enable"] = False
    cfg["calib.k.enable"] = False

    # Basic force weights so the plan run yields non-trivial results.
    cfg.update(
        {
            "ll.k.repulse": 150.0,
            "pl.k.repulse": 200.0,
            "ln.k.repulse": 180.0,
            "boundary.k.wall": 80.0,
            "anchor.k.spring": 10.0,
        }
    )

    # Viewer flags (examples should not pop UI by default)
    cfg["viz.show"] = True
    cfg["viz.field.kind"] = cfg.get("viz.field.kind", "heatmap")
    cfg["viz.field.cmap"] = cfg.get("viz.field.cmap", "viridis")
    return cfg


def build_random_schema(scene: Dict[str, Any] | None = None, cfg: Dict[str, Any] | None = None):
    """Return a trivial schema describing the example's two stages."""

    return {"stages": build_solve_plan(scene, cfg)}


def compile_solve_plan(schema: Dict[str, Any], cfg: Dict[str, Any] | None = None):
    """Compile a schema into a concrete solve plan."""

    plan = list(schema.get("stages", [])) if isinstance(schema, dict) else []
    if not plan:
        raise ValueError("empty solve_plan content in random_solve_plan example")
    return plan


def build_solve_plan(scene: Dict[str, Any] | None = None, cfg: Dict[str, Any] | None = None):
    """Construct the default two-stage solve plan used by the example."""

    return [
        {"name": "warmup_no_anchor", "scale": {"anchor.k.spring": 0.0}},
        {"name": "main_solve"},
    ]


def run_example_headless(scene: Dict[str, Any], plan, cfg: Dict[str, Any]):
    """Thin wrapper used by tests to execute the example without a viewer."""
    if not plan:
        raise ValueError("empty solve_plan content in random_solve_plan example")
    script = scene.get("scene_script") or {"steps": [{"name": "step0"}]}
    if isinstance(script, list):
        script = {"steps": script}
    return solve_scene_script(scene, script, cfg, solve_plan=plan)


def build_viz_payload(info: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt runner output for the viewer."""

    hist = info.get("history", {}) if isinstance(info, dict) else {}
    recs = hist.get("records") or hist.get("evals") or []
    frames: list[Dict[str, Any]] = []
    for r in recs:
        comps = {}
        for k, v in r.get("comps", {}).items():
            arr = np.asarray(v, float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                comps[k] = arr
        meta = dict(r.get("meta", {}))
        frames.append({"P": np.asarray(r.get("P"), float), "comps": comps, "meta": meta})

    raw_steps = hist.get("scene_steps", [])
    steps: list[Dict[str, Any]] = []
    for i, s in enumerate(raw_steps):
        rec_start = int(s.get("rec_start", 0))
        rec_end = int(s.get("rec_end", 0))
        step = {
            "id": i,
            "name": s.get("name", f"step_{i}"),
            "rec_start": rec_start,
            "rec_end": rec_end,
            "start": rec_start,
            "end": rec_end,
        }
        for k, v in s.items():
            if k not in step:
                step[k] = v
        steps.append(step)

    boundaries = [s["rec_start"] for s in steps]
    if frames:
        last_end = steps[-1]["rec_end"] if steps else len(frames)
        boundaries.append(last_end)

    def _get_idx(sid: int) -> int:
        for s in steps:
            if s.get("id") == sid:
                return int(s.get("rec_start", 0))
        return 0

    return {
        "frames": frames,
        "steps": steps,
        "actions": steps,
        "boundaries": boundaries,
        "get_frame_index_for_step": _get_idx,
        "selected_label": 0,
    }


def main():
    cfg = _build_cfg()
    scene = get_scene(
        use_random=GENERATE_NEW, cache_path=CACHE_PATH, with_scene_script=True
    )
    scene_script = scene.get("scene_script") or {"steps": [{"name": "step0"}]}
    if isinstance(scene_script, list):
        scene_script = {"steps": scene_script}
    plan = build_solve_plan(cfg)

    info = solve_scene_script(scene, scene_script, cfg, solve_plan=plan)
    print(
        f"[example] steps={len(scene_script['steps'])} frame={scene['frame_size']}"
    )
    P_final = info.get("P_final", scene.get("labels_init"))
    max_disp = float(np.abs(P_final - scene["labels_init"]).max())
    print("[random_solve_plan] labels:", P_final.shape[0], "max_disp:", f"{max_disp:.2f}")

    payload = build_viz_payload(info)

    if interactive_view and cfg.get("viz.show", False):
        lines_draw = [seg for seg in scene.get("lines", [])]
        areas_draw = [a.get("polygon") for a in scene.get("areas", [])]

        frames = payload["frames"]
        if frames:
            try:
                traj = np.stack([f["P"] for f in frames])
            except ValueError:
                traj = np.stack([scene["labels_init"], P_final])
        else:
            traj = np.stack([scene["labels_init"], P_final])

        def _force(idx=None):
            return frames[idx]["comps"] if frames else {}

        interactive_view(
            traj=traj,
            labels=scene["labels"],
            rect_wh=scene["WH"],
            points=scene["points"],
            lines=lines_draw,
            areas=areas_draw,
            W=scene["frame_size"][0],
            H=scene["frame_size"][1],
            force_getter=_force,
            field_kind=cfg.get("viz.field.kind", "heatmap"),
            field_cmap=cfg.get("viz.field.cmap", "viridis"),
            actions=payload.get("steps"),
            boundaries=payload.get("boundaries"),
        )


if __name__ == "__main__":  # pragma: no cover
    main()

