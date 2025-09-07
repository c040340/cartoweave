"""Minimal helpers to construct SolvePack for examples."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from cartoweave.config.loader import load_data_defaults
from cartoweave.config.schema import DataConfig
from cartoweave.contracts.solvepack import SolvePack

from .generate import generate_scene

__all__ = ["build_solvepack_from_config"]


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def build_solvepack_from_config(
    config: Mapping[str, Any] | str,
    seed: int | None = None,
) -> SolvePack:
    """Build a :class:`SolvePack` by merging ``config`` with ``configs/data.yaml``."""

    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    else:
        cfg = dict(config or {})

    # Merge provided data config with defaults
    default_data = load_data_defaults("configs/data.yaml").model_dump()
    user_data = cfg.get("data", {})
    gen_override = user_data.get("generate", {})
    if gen_override:
        counts = gen_override.setdefault("counts", {})
        legacy = {
            "num_points": "points",
            "num_lines": "lines",
            "num_areas": "areas",
            "total_labels": "total_labels",
        }
        for old, new in legacy.items():
            if old in gen_override:
                counts[new] = gen_override.pop(old)
        if "num_labels" in gen_override:
            n = gen_override.pop("num_labels")
            counts["points"] = n
            counts["lines"] = 0
            counts["areas"] = 0
        if "num_steps" in gen_override:
            gen_override["steps"] = gen_override.pop("num_steps")
    merged = _deep_update(default_data, user_data)
    data_cfg: DataConfig = DataConfig.model_validate(merged)

    if data_cfg.source != "generate":  # pragma: no cover - non-default path
        raise NotImplementedError("loading scenes from file is not supported")

    gen = data_cfg.generate
    assert gen is not None
    rng = np.random.default_rng(seed if seed is not None else gen.seed or 0)
    P0, labels0, active0, scene0 = generate_scene(gen, rng)

    compute_cfg = cfg.get("compute", {})
    pack = SolvePack(
        L=len(labels0),
        P0=[(float(x), float(y)) for x, y in P0],
        labels0=labels0,
        active0=active0.tolist(),
        scene0=scene0,
        cfg={"compute": compute_cfg},
    )
    return pack
