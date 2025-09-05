"""Scene caching helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os

import numpy as np

from .script import build_scene_script


def save_scene(path: str | Path, payload: Dict) -> None:
    """Save ``payload`` to ``path`` using ``np.savez_compressed``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=np.array(payload, dtype=object))


def load_scene(path: str | Path) -> Dict:
    """Load a scene payload saved with :func:`save_scene`."""
    obj = np.load(path, allow_pickle=True)["data"].item()
    return obj


def ensure_script(payload: Dict) -> Dict:
    """Ensure ``payload`` contains a scene script."""
    if "script" not in payload:
        n_labels = payload.get("labels_init", np.zeros((0, 2))).shape[0]
        n_steps = int(payload.get("n_steps", max(1, n_labels)))
        rng = np.random.default_rng()
        payload["script"] = build_scene_script(rng, n_labels, n_steps)
    return payload

