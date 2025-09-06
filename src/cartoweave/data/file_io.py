from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["load_scene", "save_scene"]


def save_scene(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=np.array(payload, dtype=object))


def load_scene(path: str | Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True)["data"].item()
