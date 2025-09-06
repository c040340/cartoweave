from __future__ import annotations

import numpy as np

from ..sampling.helpers import frame_metrics, inset_rect

__all__ = ["generate_points"]


def generate_points(params, rng: np.random.Generator) -> np.ndarray:
    frame_size = (params["frame"]["width"], params["frame"]["height"])
    diag, _ = frame_metrics(frame_size)
    inset = params["random"]["route_gen"]["inset_margin_scale"] * diag
    rect = inset_rect(frame_size, inset)
    n_points = int(params["counts"]["points"])
    if n_points <= 0:
        return np.zeros((0, 2), float)
    return rng.uniform([rect[0], rect[1]], [rect[2], rect[3]], size=(n_points, 2))
