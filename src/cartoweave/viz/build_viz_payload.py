from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

from cartoweave.utils.logging import logger
from cartoweave.utils.numerics import (
    is_finite_array,
    sanitize_array,
    repair_nonfinite_rows,
)


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
    N = len(labels_all)
    labels_init = np.asarray(scene.get("labels_init", np.zeros((N, 2))), float).reshape(N, 2)
    if N == 0 and recs:
        first_P = np.asarray(recs[0].get("P"), float)
        if first_P.ndim == 2 and first_P.shape[1] == 2:
            N = first_P.shape[0]
            labels_all = [{}] * N
            labels_init = np.asarray(scene.get("labels_init", first_P), float).reshape(N, 2)

    frames: List[Dict[str, Any]] = []
    prev_full: np.ndarray | None = None
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

        act_viz = (
            list(steps_raw[step_idx].get("active_ids_viz", []))
            if steps_raw
            else list(range(N))
        )
        act_solver = (
            list(steps_raw[step_idx].get("active_ids_solver", act_viz))
            if steps_raw
            else list(range(N))
        )

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

        act_viz = list(steps_raw[step_idx].get("active_ids_viz", [])) if steps_raw else list(range(len(labels_all)))
        act_sol = list(steps_raw[step_idx].get("active_ids_solver", [])) if steps_raw else list(range(len(labels_all)))

        N = len(labels_all)

        # --- previous full P for repair fallback ---
        if frames:
            prev_full = np.asarray(frames[-1]["P"], float)
        else:
            # 初始全量位置：优先 labels_init_all，其次 labels_init，再不济 zeros
            L0 = np.asarray(scene.get("labels_init_all", scene.get("labels_init", np.zeros((N, 2)))), float)
            prev_full = L0

        # --- current solver-subset P (shape = (len(act_sol), 2)) ---
        P_sub = np.asarray(r.get("P"), float)

        # --- repair non-finite rows in the subset, using prev_full / init ---
        if not is_finite_array(P_sub):
            P_sub = P_sub.copy()
            for k, idx in enumerate(act_sol):
                row = np.asarray(P_sub[k], float)
                if not np.isfinite(row).all():
                    # 回填为上一帧该 label 的位置；若无上一帧则用初始
                    fallback = prev_full[idx] if (0 <= idx < prev_full.shape[0]) else np.zeros((2,), float)
                    logger.warning("repair NaN/Inf in P: frame=%d label=%d -> fallback", t, idx)
                    P_sub[k] = fallback

        # --- expand subset back to full-length array ---
        P_full = np.array(prev_full, dtype=float, copy=True)
        for k, idx in enumerate(act_sol):
            if 0 <= idx < N:
                P_full[idx] = P_sub[idx]

        frame = {
            "P": P_full,
            "comps": comps,  # 注意：comps 仍是子集（每行对齐 act_sol 顺序）
            "meta": meta,
            "sources_for_step": {"points": pts, "lines": lns, "areas": ars},
            "active_ids_viz": act_viz,
            "active_ids_solver": act_sol,  # 记录下来用于调试/防御
        }
        frame["active_ids_viz"] = act_viz
        frame["active_ids_solver"] = act_solver
        prev_full = P_full
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
