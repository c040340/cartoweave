from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


def build_viz_payload(info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert solver output into a viewer-friendly payload.

    Each frame records the per-iteration positions and force components while
    attaching the active visualization mask derived from ``history.scene_steps``.
    """

    hist = info.get("history", {}) if isinstance(info, dict) else {}
    recs = hist.get("records") or hist.get("evals") or []
    steps_raw = hist.get("scene_steps", [])

    frames: List[Dict[str, Any]] = []
    for t, r in enumerate(recs):
        comps: Dict[str, np.ndarray] = {}
        for k, v in r.get("comps", {}).items():
            arr = np.asarray(v, float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                comps[k] = arr
        frame = {
            "P": np.asarray(r.get("P"), float),
            "comps": comps,
            "meta": dict(r.get("meta", {})),
        }
        # attach per-frame viz mask
        if steps_raw:
            step_idx = 0
            for i, s in enumerate(steps_raw):
                if int(s.get("rec_start", 0)) <= t < int(s.get("rec_end", 0)):
                    step_idx = i
                    break
            frame["active_ids_viz"] = list(steps_raw[step_idx].get("active_ids_viz", []))
        else:
            frame["active_ids_viz"] = list(range(frame["P"].shape[0]))
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
