"""Minimal end-to-end example (config-compatible, T1–T7).

- Builds a tiny scene with three labels (point/line/area anchors).
- Provides a hand-rolled configuration dictionary.
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
from typing import Dict, Any, List
import os
import numpy as np
from cartoweave.labels import anchor_xy, init_position
from cartoweave.viz.build_viz_payload import build_viz_payload  # noqa: E402

from cartoweave.viz.backend import use_compatible_backend
use_compatible_backend()

from cartoweave.api import solve_frame  # noqa: E402
from cartoweave.config.loader import load_configs, print_effective_config  # noqa: E402
from cartoweave.utils.dict_merge import deep_update  # noqa: E402
from cartoweave.utils.logging import logger, configure_logging  # noqa: E402

try:
    # optional viz
    from cartoweave.viz.view import interactive_view
except Exception:
    interactive_view = None  # headless environments are fine

def _make_scene(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_rand = cfg.get("data", {}).get("random", {})
    frame = data_rand.get("frame", {})
    W = float(frame.get("width", 800.0))
    H = float(frame.get("height", 600.0))
    frame_size = (W, H)

    # Anchors
    points = np.array([[150.0, 120.0]], dtype=float)          # one point
    lines  = [np.array([[450.0,  80.0],[650.0, 140.0]], float)]  # one segment
    areas  = [{"polygon": np.array([[380.0, 260.0],
                                    [520.0, 260.0],
                                    [520.0, 360.0],
                                    [380.0, 360.0]], float)}]    # one rectangle

    labels = [
        {"anchor_kind": "point", "anchor_index": 0},
        {"anchor_kind": "line",  "anchor_index": 0},
        {"anchor_kind": "area",  "anchor_index": 0},
    ]

    data_geo = {"points": points, "lines": lines, "areas": [areas[0]["polygon"]]}
    anchors_list: List[np.ndarray] = []
    init_list: List[np.ndarray] = []
    for lab in labels:
        qx, qy, meta = anchor_xy(lab["anchor_kind"], lab["anchor_index"], data_geo, frame_size, with_meta=True)
        anc = np.array([qx, qy], float)
        anchors_list.append(anc)
        init_list.append(init_position(lab["anchor_kind"], anc, frame_size, meta=meta))
    anchors = np.vstack(anchors_list)
    P0 = np.vstack(init_list)
    WH = np.array([[60.0, 24.0]] * len(labels), float)

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

def main():
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, help="WIDTHxHEIGHT override")
    parser.add_argument("--override", type=str, help="JSON blob of overrides")
    args = parser.parse_args()

    configure_logging()

    base = os.path.join(os.path.dirname(__file__), "..")
    cfg = load_configs(
        internals_path=os.path.join(base, "configs/solver.internals.yaml"),
        tuning_path=os.path.join(base, "configs/solver.tuning.yaml"),
        public_path=os.path.join(base, "configs/solver.public.yaml"),
        viz_path=os.path.join(base, "configs/viz.yaml"),
    )
    overrides: Dict[str, Any] = {}
    if args.frame:
        try:
            w, h = map(int, args.frame.lower().split("x"))
            overrides = deep_update(overrides, {"data": {"random": {"frame": {"width": w, "height": h}}}})
        except Exception:  # pragma: no cover - CLI parsing error
            pass
    if args.override:
        try:
            overrides = deep_update(overrides, json.loads(args.override))
        except Exception:  # pragma: no cover - bad JSON
            pass
    cfg = deep_update(
        cfg,
        {
            "viz": {"show": False, "field": {"kind": "heatmap", "cmap": "viridis"}},
            "engine": {"max_iter_hint": 200},
        },
    )
    if overrides:
        def _flatten(d, prefix=""):
            for k, v in d.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    yield from _flatten(v, path)
                else:
                    yield path, v
        for path, val in _flatten(overrides):
            prev = cfg
            for key in path.split("."):
                if isinstance(prev, dict) and key in prev:
                    prev = prev[key]
                else:
                    prev = None
                    break
            logger.info("[override] %s: %s → %s", path, prev, val)
        cfg = deep_update(cfg, overrides)

    print_effective_config(cfg)

    viz = deep_update(cfg.get("viz", {}), {})
    viz_override = {"layout": {"anchor_marker_size": 8.0}} if os.getenv("DEMO_BIG_ANCHOR") else {}
    viz_eff = deep_update(viz, viz_override)

    logger.info(
        "configs loaded config=%s viz=%s run=%s anchor_marker_size=%.1f",
        "configs/solver.*.yaml",
        "configs/viz.yaml",
        "<memory>",
        float(viz_eff.get("layout", {}).get("anchor_marker_size", 0.0)),
    )

    scene = _make_scene(cfg)

    P_opt, info = solve_frame(scene, cfg, mode="lbfgs")
    print("[minimal_fit] done. final positions shape:", P_opt.shape)

    payload = build_viz_payload(info)

    # Optional interactive viewer
    if interactive_view and cfg.get("viz", {}).get("show", False):
        lines_draw = [seg for seg in scene["lines"]]
        areas_draw = [a["polygon"] for a in scene["areas"]]

        frames = payload["frames"]
        traj = np.stack([f["P"] for f in frames]) if frames else np.asarray([P_opt])

        def _force(idx=None):
            return frames[idx]["comps"] if frames else {}

        def _active(idx=None):
            return frames[idx].get("active_ids_viz", list(range(len(scene["labels"])))) if frames else list(range(len(scene["labels"])))

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
            active_getter=_active,
            field_kind=cfg.get("viz.field.kind", "heatmap"),
            field_cmap=cfg.get("viz.field.cmap", "viridis"),
            actions=payload.get("steps"),
            boundaries=payload.get("boundaries"),
            viz=viz_eff,
        )

if __name__ == "__main__":
    main()
