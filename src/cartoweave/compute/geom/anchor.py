from cartoweave.labels import anchor_xy as _anchor_xy
import numpy as np

__all__ = ["anchor_xy"]


def anchor_xy(anchor: dict, scene: dict):
    if anchor is None:
        return None
    kind = anchor.get("kind") or anchor.get("anchor_kind")
    idx = anchor.get("index") if "index" in anchor else anchor.get("anchor_index")
    if kind is None or idx is None:
        return None
    frame = scene.get("frame_size", (0, 0))
    x, y = _anchor_xy(kind, int(idx), scene, frame)
    return np.array([x, y], float)
