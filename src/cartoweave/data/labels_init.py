from __future__ import annotations

from typing import Any

import numpy as np

from cartoweave.labels import anchor_xy, init_position

__all__ = ["compute_P0"]


def compute_P0(payload: dict[str, Any], rng: np.random.Generator) -> np.ndarray:  # noqa: N802
    """Compute initial label positions ``P0`` for ``payload``."""
    labels = payload.get("labels", [])
    frame = payload.get("frame_size", (1, 1))
    n_labels = len(labels)
    p0 = np.zeros((n_labels, 2), dtype=float)
    for i, lbl in enumerate(labels):
        kind = lbl.get("anchor_kind", "none")
        idx = int(lbl.get("anchor_index", -1))
        if kind != "none":
            ax, ay, meta = anchor_xy(kind, idx, payload, frame, with_meta=True)
            p0[i] = init_position(
                kind,
                (ax, ay),
                frame,
                prev=None,
                mode=lbl.get("mode"),
                locked=False,
                meta=meta,
            )
        else:
            cx, cy = frame[0] / 2.0, frame[1] / 2.0
            scale = max(frame) / 6.0
            p0[i] = rng.normal([cx, cy], scale)
    return p0
