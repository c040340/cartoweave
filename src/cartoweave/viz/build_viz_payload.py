from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

from cartoweave.utils.logging import logger
from cartoweave.utils.numerics import is_finite_array, sanitize_array


def build_viz_payload(info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert solver output into a viewer-friendly payload.

    Each frame records the per-iteration positions and force components while
    attaching the active visualization mask derived from ``history.scene_steps``.
    """

    hist = info.get("history", {}) if isinstance(info, dict) else {}
    recs = hist.get("records") or hist.get("evals") or []
    steps_raw = hist.get("scene_steps", [])
    scene = info.get("scene", {}) if isinstance(info, dict) else {}
    pts_all = np.asarray(scene.get("points", np.zeros((0, 2))), float).reshape(-1, 2)
    lns_all = scene.get("lines", [])
    ars_all = scene.get("areas", [])
    labels_all = scene.get("labels", [])

    frames: List[Dict[str, Any]] = []
    for t, r in enumerate(recs):
        comps: Dict[str, np.ndarray] = {}
        for k, v in r.get("comps", {}).items():
            arr = np.asarray(v, float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                if not is_finite_array(arr):
                    logger.warning("sanitize payload: frame=%d term=%s", t, k)
                    arr = sanitize_array(arr)
                comps[k] = arr
        meta = dict(r.get("meta", {}))

        # determine which step the record belongs to
        step_idx = 0
        if steps_raw:
            for i, s in enumerate(steps_raw):
                if int(s.get("rec_start", 0)) <= t < int(s.get("rec_end", 0)):
                    step_idx = i
                    break

        # pull geometry sources for the current step
        src = meta.get("sources") or {}
        pts_val = src.get("points")
        if pts_val is None:
            pts_val = src.get("points_xy")
        pts = np.asarray(pts_val, float).reshape(-1, 2) if pts_val is not None else None
        if pts is not None and not is_finite_array(pts):
            logger.warning("sanitize payload: frame=%d sources.points", t)
            pts = sanitize_array(pts)

        lns_raw = src.get("lines")
        if lns_raw is None:
            lns_raw = src.get("lines_xy") or []
        lns = []
        for seg in lns_raw:
            arr = np.asarray(seg, float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                if not is_finite_array(arr):
                    logger.warning("sanitize payload: frame=%d sources.line", t)
                    arr = sanitize_array(arr)
                lns.append(arr)
        if len(lns) == 0:
            lns = None

        ars_raw = src.get("areas")
        if ars_raw is None:
            ars_raw = src.get("areas_xy") or []
        ars = []
        for poly in ars_raw:
            arr = np.asarray(poly, float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                if not is_finite_array(arr):
                    logger.warning("sanitize payload: frame=%d sources.area", t)
                    arr = sanitize_array(arr)
                ars.append(arr)
        if len(ars) == 0:
            ars = None

        act_viz = list(steps_raw[step_idx].get("active_ids_viz", [])) if steps_raw else list(range(len(labels_all)))
        if pts is None:
            pts_sel = []
            for i in act_viz:
                lab = labels_all[i] if i < len(labels_all) else {}
                if lab.get("anchor_kind") == "point":
                    ai = lab.get("anchor_index")
                    if isinstance(ai, int) and 0 <= ai < len(pts_all):
                        pts_sel.append(pts_all[ai])
            pts = np.asarray(pts_sel, float)
        if lns is None:
            lns = []
            for i in act_viz:
                lab = labels_all[i] if i < len(labels_all) else {}
                if lab.get("anchor_kind") == "line":
                    ai = lab.get("anchor_index")
                    if isinstance(ai, int) and 0 <= ai < len(lns_all):
                        lns.append(np.asarray(lns_all[ai], float))
        if ars is None:
            ars = []
            for i in act_viz:
                lab = labels_all[i] if i < len(labels_all) else {}
                if lab.get("anchor_kind") == "area":
                    ai = lab.get("anchor_index")
                    if isinstance(ai, int) and 0 <= ai < len(ars_all):
                        poly = ars_all[ai]
                        if isinstance(poly, dict):
                            poly = poly.get("polygon")
                        ars.append(np.asarray(poly, float))

        P_arr = np.asarray(r.get("P"), float)
        if not is_finite_array(P_arr):
            logger.warning("sanitize payload: frame=%d field=P", t)
            P_arr = sanitize_array(P_arr)
        frame = {
            "P": P_arr,
            "comps": comps,
            "meta": meta,
            "sources_for_step": {"points": pts, "lines": lns, "areas": ars},
        }

        if steps_raw:
            frame["active_ids_viz"] = list(steps_raw[step_idx].get("active_ids_viz", []))
        else:
            frame["active_ids_viz"] = list(range(frame["P"].shape[0]))

        logger.info(
            "[viz] step %d: pts=%d lines=%d areas=%d",
            step_idx,
            len(pts),
            len(lns),
            len(ars),
        )

        frames.append(frame)

    steps: List[Dict[str, Any]] = []
    for i, s in enumerate(steps_raw):
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

    if frames:
        first_mask = frames[0].get("active_ids_viz", [])
        print(f"[viz] frame=0 active_viz={len(first_mask)} total={frames[0]['P'].shape[0]}")

    return {
        "frames": frames,
        "steps": steps,
        "actions": steps,
        "boundaries": boundaries,
        "get_frame_index_for_step": _get_idx,
        "selected_label": 0,
    }
