from __future__ import annotations

from typing import Any

import numpy as np

from ..textspec import label_specs_for_len
from . import areas as A  # noqa: N812
from . import lines as L  # noqa: N812
from . import points as P  # noqa: N812

__all__ = ["generate_payload", "assign_labels", "measure_WH"]


def generate_payload(params: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    frame_size = (params["frame"]["width"], params["frame"]["height"])
    points = P.generate_points(params, rng)
    lines = L.generate_lines(params, rng)
    areas = A.generate_areas(params, rng)
    geom = {"points": points, "lines": lines, "areas": areas, "frame_size": frame_size}
    anchors_cfg = params.get("anchors", {})
    labels = assign_labels(
        geom,
        params["counts"]["labels"],
        anchors_cfg.get("policy", "auto"),
        anchors_cfg.get("modes", {}),
        rng,
    )
    wh = measure_WH(len(labels), rng)
    payload = {
        "frame_size": frame_size,
        "points": points,
        "lines": lines,
        "areas": areas,
        "labels": labels,
        "WH": wh,
    }
    if params.get("steps"):
        payload["steps"] = params["steps"]
    return payload


def assign_labels(geometry: dict[str, Any], n_labels: int, anchors_policy, anchor_modes, rng: np.random.Generator):
    pts = np.asarray(geometry.get("points", []))
    lines = geometry.get("lines", [])
    areas = geometry.get("areas", [])
    counts = {"point": len(pts), "line": len(lines), "area": len(areas)}

    if isinstance(anchors_policy, str):
        if anchors_policy == "auto":
            weights = {"point": 1.0, "line": 1.0, "area": 1.0, "none": 1.0}
        elif anchors_policy == "none":
            weights = {"none": 1.0}
        else:
            raise ValueError(f"unknown anchors policy {anchors_policy!r}")
    else:
        weights = {k: float(v) for k, v in anchors_policy.items()}

    kinds_all = ["point", "line", "area", "none"]
    labels = []
    for _ in range(n_labels):
        kinds = []
        probs = []
        for k in kinds_all:
            if k == "none" or counts.get(k, 0) > 0:
                w = weights.get(k, 0.0)
                if w > 0:
                    kinds.append(k)
                    probs.append(w)
        if not kinds:
            kind = "none"
        else:
            probs_arr = np.array(probs, dtype=float)
            probs_arr /= probs_arr.sum()
            kind = rng.choice(kinds, p=probs_arr)
        if kind == "none":
            idx = -1
        else:
            idx = int(rng.integers(counts[kind]))
        label = {"anchor_kind": kind, "anchor_index": idx}
        if kind in ("line", "area"):
            mode = anchor_modes.get(kind)
            if mode:
                label["mode"] = mode
        labels.append(label)
    return labels


def measure_WH(n_labels: int, rng: np.random.Generator) -> np.ndarray:  # noqa: N802
    spec = label_specs_for_len(10)
    return np.array([[spec.single_px, 24.0] for _ in range(n_labels)], dtype=float)
