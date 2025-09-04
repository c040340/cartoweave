"""Random scene + short timeline (config-compatible, T1–T7).

- Generates or loads a cached random scene.
- Loads layered config and fills calibration strategy (fill-only).
- Keeps calibration gates OFF for stable example output.
- Runs a short 2-step timeline using the legacy orchestrator (auto-carries P).
"""

# Root cause:
#     This example previously bypassed the dedicated timeline runner and invoked
#     the solver directly, meaning no per-action metadata was recorded.  The
#     runner is now exposed via ``run_example_headless`` and the default
#     timeline builder so tests and the viewer receive proper action ids.

from __future__ import annotations
from typing import Dict, Any
import os

from cartoweave.viz.backend import use_compatible_backend
use_compatible_backend()

from cartoweave.config.layering import load_base_cfg, apply_calib_profile, apply_shape_profile
from cartoweave.data.random import generate_scene, save_scene, load_scene
from cartoweave.orchestrators.timeline_runner import run_timeline
import numpy as np

try:  # optional viewer
    from cartoweave.viz.view import interactive_view
except Exception:  # pragma: no cover - viewer not installed
    interactive_view = None

_CACHE = os.environ.get("CARTOWEAVE_EXAMPLE_CACHE", "examples/_scene_cache.npz")

def _get_scene() -> Dict[str, Any]:
    """Load cached scene if valid; otherwise regenerate."""
    if os.path.exists(_CACHE):
        data = load_scene(_CACHE)
        required = {"labels_init", "WH", "anchors", "labels"}
        if required.issubset(data.keys()):
            return data
    data = generate_scene(canvas_size=(1080, 1920),
                          n_points=12, n_lines=2, n_areas=1, seed=42)
    save_scene(data, _CACHE)
    return data

def _build_cfg() -> Dict[str, Any]:
    cfg = load_base_cfg()
    apply_calib_profile(cfg, cfg.get("calib.k.profile", "default"), fill_only=True)

    # Apply one shape profile so base forces exist but keep gates OFF for
    # deterministic lightweight runs.
    apply_shape_profile(cfg, cfg.get("calib.shape.name", "default"), enable=True)
    cfg["calib.shape.enable"] = False
    cfg["calib.k.enable"]     = False

    # Basic force weights so the timeline run yields non-trivial results.
    cfg.update({
        "ll.k.repulse": 150.0,
        "pl.k.repulse": 200.0,
        "ln.k.repulse": 180.0,
        "boundary.k.wall": 80.0,
        "anchor.k.spring": 10.0,
    })

    # Viewer flags (examples should not pop UI by default)
    cfg["viz.show"] = True
    cfg["viz.field.kind"] = cfg.get("viz.field.kind", "heatmap")
    cfg["viz.field.cmap"] = cfg.get("viz.field.cmap", "viridis")
    return cfg

def build_random_timeline(scene: Dict[str, Any] | None = None, cfg: Dict[str, Any] | None = None):
    """Construct the default two-action timeline used by the example.

    Parameters are accepted for API symmetry with other builders but are not
    currently inspected.
    """

    return [
        {"name": "warmup_no_anchor", "scale": {"anchor.k.spring": 0.0}},
        {"name": "main_solve"},
    ]


def run_example_headless(scene: Dict[str, Any], timeline, cfg: Dict[str, Any]):
    """Thin wrapper used by tests to execute the example without a viewer."""
    return run_timeline(scene, timeline, cfg)

def build_viz_payload(info: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt timeline runner output for the viewer."""

    hist = info.get("history", {}) if isinstance(info, dict) else {}
    recs = hist.get("records") or hist.get("evals") or []
    frames: list[Dict[str, Any]] = []
    action_segments: list[Dict[str, Any]] = []
    last_aid = None
    last_name = None
    seg_start = 0
    for i, r in enumerate(recs):
        comps = {}
        for k, v in r.get("comps", {}).items():
            arr = np.asarray(v, float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                comps[k] = arr
        meta = dict(r.get("meta", {}))
        frame = {"P": np.asarray(r.get("P"), float), "comps": comps, "meta": meta}
        aid = meta.get("action_id")
        aname = meta.get("action_name", f"action_{aid}") if aid is not None else None
        if aid is not None and aid != last_aid:
            if last_aid is not None:
                action_segments.append({"id": last_aid, "name": last_name, "start": seg_start, "end": i})
            last_aid = aid
            last_name = aname
            seg_start = i
        frames.append(frame)
    if last_aid is not None:
        action_segments.append({"id": last_aid, "name": last_name, "start": seg_start, "end": len(recs)})

    boundaries = [seg["start"] for seg in action_segments]
    if frames:
        boundaries.append(len(frames))

    def _get_idx(aid: int) -> int:
        for seg in action_segments:
            if seg.get("id") == aid:
                return int(seg.get("start", 0))
        return 0

    return {
        "frames": frames,
        "action_segments": action_segments,
        "actions": action_segments,
        "boundaries": boundaries,
        "get_frame_index_for_action": _get_idx,
        "selected_label": 0,
    }


def main():
    scene = _get_scene()
    cfg   = _build_cfg()
    schedule = build_random_timeline(scene, cfg)

    result = run_example_headless(scene, schedule, cfg)
    P_final = result["P_final"]
    max_disp = float(np.abs(P_final - scene["labels_init"]).max())
    print("[random_timeline] steps:", len(schedule), "labels:", P_final.shape[0], "max_disp:", f"{max_disp:.2f}")

    payload = build_viz_payload(result)

    if interactive_view and cfg.get("viz.show", False):
        lines_draw = [seg for seg in scene["lines"]]
        areas_draw = [a["polygon"] for a in scene["areas"]]

        frames = payload["frames"]
        traj = np.stack([f["P"] for f in frames]) if frames else np.stack([scene["labels_init"], P_final])

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
            actions=payload.get("actions"),
            boundaries=payload.get("boundaries"),
        )

if __name__ == "__main__":
    main()
