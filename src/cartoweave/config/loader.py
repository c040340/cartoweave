"""Compute configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from cartoweave.utils.dict_merge import deep_update
from .schema import Compute

__all__ = ["load_compute_config", "load_configs"]


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Top-level YAML at {path} must be a mapping")
        return data


def _ensure_only_compute(d: Dict[str, Any]) -> None:
    extra = set(d.keys()) - {"compute"}
    if extra:
        raise ValueError(f"unexpected top-level keys: {sorted(extra)}")


def load_compute_config(
    *,
    internals_path: str = "configs/compute.internals.yaml",
    tuning_path: str = "configs/compute.tuning.yaml",
    public_path: str = "configs/compute.public.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load compute configuration and return ``{"compute": {...}}``."""

    internals = _read_yaml(internals_path)
    tuning = _read_yaml(tuning_path)
    public = _read_yaml(public_path)

    for d in (internals, tuning, public):
        _ensure_only_compute(d)

    cfg: Dict[str, Any] = {}
    cfg = deep_update(cfg, internals.get("compute", {}))
    cfg = deep_update(cfg, public.get("compute", {}))
    cfg = deep_update(cfg, tuning.get("compute", {}))

    if overrides:
        _ensure_only_compute(overrides)
        cfg = deep_update(cfg, overrides.get("compute", {}))

    model = Compute.model_validate(cfg)
    return {"compute": model.model_dump()}


# Backwards-compatible alias
load_configs = load_compute_config
