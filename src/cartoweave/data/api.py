"""Minimal helpers to construct SolvePack for examples."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Dict

import numpy as np
import yaml

from cartoweave.contracts.solvepack import SolvePack, Scene, Label, Anchor

__all__ = ["build_solvepack_from_config"]


def build_solvepack_from_config(config: Mapping[str, Any] | str, seed: int | None = None) -> SolvePack:
    """Construct a very small :class:`SolvePack` for demonstration purposes."""
    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    else:
        cfg = dict(config or {})

    gen = cfg.get("data", {}).get("generate", {})
    frame_size = tuple(gen.get("frame_size", (200.0, 100.0)))
    N = int(gen.get("num_labels", 1))
    rng = np.random.default_rng(seed or 0)
    P0 = rng.uniform([0.0, 0.0], frame_size, size=(N, 2))
    active0 = np.zeros(N, dtype=bool)
    labels0 = [
        Label(id=i, kind="point", anchor=Anchor(mode="xy", xy=(float(x), float(y))))
        for i, (x, y) in enumerate(P0)
    ]
    scene = Scene(frame_size=frame_size)
    compute_cfg = cfg.get("compute", {})
    pack = SolvePack(
        L=N,
        P0=[(float(x), float(y)) for x, y in P0],
        labels0=labels0,
        active0=active0.tolist(),
        scene0=scene,
        cfg={"compute": compute_cfg},
    )
    return pack
