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
from typing import Dict, Any, Tuple, List
import numpy as np
from cartoweave.labels import anchor_xy, init_position
from cartoweave.viz.build_viz_payload import build_viz_payload  # noqa: E402

from cartoweave.viz.backend import use_compatible_backend
use_compatible_backend()

from cartoweave.api import solve_frame

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

def _build_cfg() -> Dict[str, Any]:
    """Return a minimal configuration dictionary for the example.

    The previous version depended on a layered configuration package.  To
    keep the example lightweight and self-contained, we now construct the
    few required parameters directly.
    """

    return {
        "ll.k.repulse": 150.0,
        "pl.k.repulse": 200.0,
        "ln.k.repulse": 180.0,
        "boundary.k.wall": 80.0,
        "anchor.k.spring": 10.0,
        "viz.show": False,
        "viz.field.kind": "heatmap",
        "viz.field.cmap": "viridis",
        "engine.max_iter_hint": 200,
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
        )

if __name__ == "__main__":
    main()
