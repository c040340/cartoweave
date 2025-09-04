from __future__ import annotations
from typing import TypedDict, Tuple
import numpy as np

class SceneData(TypedDict, total=False):
    frame: int
    frame_size: Tuple[int, int]   # (W,H)
    points: np.ndarray            # (Np,2)
    lines: np.ndarray             # (Nl,2,2) polylines
    areas: np.ndarray             # (Na, M) placeholder packed polygon
    labels_init: np.ndarray       # (N,2) optional
    movable_idx: np.ndarray       # (K,) int indices of movable labels
