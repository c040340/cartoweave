"""Minimal end-to-end example (config-compatible, T1–T7).

- Builds a tiny scene with three labels (point/line/area anchors).
- Loads config via layered presets.
- Fills calibration strategy (fill-only) but keeps all calibration gates OFF by default.
- Solves one frame with LBFGS and (optionally) opens the viewer if cfg says so.

Phase A diagnostics:
- ``build_viz_payload`` reads per-iteration data from ``info['history']['records']``.
- ``cartoweave.viz.view`` builds the action slider from ``boundaries`` and wires it via
  ``_on_action`` to update the iteration slider.
- The right force panel is plotted by ``draw_force_panel`` inside ``interactive_view``.
- Missing action→frame mapping and invalid force dictionaries (entries not shaped
  ``(N,2)``) left the action bar idle and the right panel blank.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np

from cartoweave.viz.backend import use_compatible_backend
use_compatible_backend()

from cartoweave.api import solve_frame
from cartoweave.config.layering import load_base_cfg, apply_calib_profile, apply_shape_profile

try:
    # optional viz
    from cartoweave.viz.view import interactive_view
except Exception:
    interactive_view = None  # headless environments are fine

def _make_scene() -> Dict[str, Any]:
    # Canvas
    W, H = 800.0, 600.0
    frame_size = (W, H)

    # Anchors
    points = np.array([[150.0, 120.0]], dtype=float)          # one point
    lines  = [np.array([[450.0,  80.0],[650.0, 140.0]], float)]  # one segment
    areas  = [{"polygon": np.array([[380.0, 260.0],
                                    [520.0, 260.0],
                                    [520.0, 360.0],
                                    [380.0, 360.0]], float)}]    # one rectangle

    # Labels (3) with initial positions and sizes
    P0 = np.array([[180.0, 110.0],
                   [460.0, 110.0],
                   [420.0, 310.0]], dtype=float)
    WH = np.array([[60.0, 24.0],
                   [60.0, 24.0],
                   [60.0, 24.0]], dtype=float)

    labels = [
        {"anchor_kind": "point", "anchor_index": 0},
        {"anchor_kind": "line",  "anchor_index": 0},
        {"anchor_kind": "area",  "anchor_index": 0},
    ]

    anchors = np.vstack([
        points[0],
        lines[0].mean(axis=0),
        areas[0]["polygon"].mean(axis=0),
    ])

    return dict(
        frame=0,
        frame_size=frame_size,
        points=points,
        lines=lines,
        areas=areas,
        labels_init=P0,
        WH=WH,
        labels=labels,
        anchors=anchors,
        # Some viewers expect these size bounds; add conservative defaults:
        WH_min=np.minimum(WH, WH),  # placeholder
        WH_max=np.maximum(WH, WH),  # placeholder
    )

def _build_cfg() -> Dict[str, Any]:
    cfg = load_base_cfg()

    # Fill calibration strategy dicts (target_rel/trigger/quantiles/limits)
    # but DO NOT turn calibration on in the example:
    apply_calib_profile(cfg, cfg.get("calib.k.profile", "default"), fill_only=True)

    # Apply a shape profile once so default forces are present; keep dynamic
    # calibration gates off for deterministic, lightweight examples.
    apply_shape_profile(cfg, cfg.get("calib.shape.name", "default"), enable=True)
    cfg["calib.shape.enable"] = False
    cfg["calib.k.enable"]     = False

    # Seed basic force weights so the solver actually moves labels.
    cfg.update({
        "ll.k.repulse": 150.0,
        "pl.k.repulse": 200.0,
        "ln.k.repulse": 180.0,
        "boundary.k.wall": 80.0,
        "anchor.k.spring": 10.0,
    })

    # Optional viewer knobs (safe defaults). If you don’t use viewer, ignore these.
    cfg["viz.show"] = True
    cfg["viz.field.kind"] = cfg.get("viz.field.kind", "heatmap")   # "3d" | "heatmap"
    cfg["viz.field.cmap"] = cfg.get("viz.field.cmap", "viridis")

    # Typical solver hint
    cfg["engine.max_iter_hint"] = cfg.get("engine.max_iter_hint", 200)
    return cfg


def build_viz_payload(info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert solver ``info`` into a viewer-friendly payload."""

    hist = info.get("history", {}) if isinstance(info, dict) else {}
    recs = hist.get("records") or hist.get("evals") or []
    frames: List[Dict[str, Any]] = []
    action_segments: List[Dict[str, Any]] = []
    last_aid = None
    last_name = None
    seg_start = 0
    for i, r in enumerate(recs):
        comps = {}
        for k, v in r.get("comps", {}).items():
            arr = np.asarray(v, float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                comps[k] = arr
        frame: Dict[str, Any] = {"P": np.asarray(r.get("P"), float), "comps": comps}
        meta = r.get("meta") or {}
        aid = meta.get("action_id")
        aname = meta.get("action_name", f"action_{aid}") if aid is not None else None
        if aid is not None and aid != last_aid:
            if last_aid is not None:
                action_segments.append({"id": last_aid, "name": last_name, "start": seg_start, "end": i})
            last_aid = aid
            last_name = aname
            seg_start = i
        if isinstance(meta, dict):
            frame["meta"] = dict(meta)
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
        "actions": action_segments,  # compat alias
        "boundaries": boundaries,
        "get_frame_index_for_action": _get_idx,
        "selected_label": 0,
    }

def main():
    scene = _make_scene()
    cfg   = _build_cfg()

    P_opt, info = solve_frame(scene, cfg, mode="lbfgs")
    print("[minimal_fit] done. final positions shape:", P_opt.shape)

    payload = build_viz_payload(info)

    # Optional interactive viewer
    if interactive_view and cfg.get("viz.show", False):
        lines_draw = [seg for seg in scene["lines"]]
        areas_draw = [a["polygon"] for a in scene["areas"]]

        frames = payload["frames"]
        traj = np.stack([f["P"] for f in frames]) if frames else np.asarray([P_opt])

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
