"""Compute configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from cartoweave.utils.dict_merge import deep_update
from .schema import Compute, ProfileData, DataConfig

__all__ = ["load_compute_config", "load_configs", "load_data_defaults"]


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


def _reject_legacy_weights(cfg: dict) -> None:
    if (
        isinstance(cfg, dict)
        and "compute" in cfg
        and isinstance(cfg["compute"], dict)
        and "weights" in cfg["compute"]
    ):
        raise ValueError(
            "Legacy 'compute.weights' is no longer supported. Use "
            "'compute.public.forces.<term>.{enable,k}' instead."
        )


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

    passes = cfg.get("passes")
    if isinstance(passes, dict) and "calibration" in passes:
        tp = cfg.setdefault("tuning_passes", {})
        if not isinstance(tp, dict):
            tp = {}
            cfg["tuning_passes"] = tp
        tp["calibration"] = passes.pop("calibration")

    _reject_legacy_weights({"compute": cfg})

    model = Compute.model_validate(cfg)
    return {"compute": model.model_dump()}


# ---------------------------------------------------------------------------
# Data configuration loader
# ---------------------------------------------------------------------------


def load_data_defaults(path: str = "configs/data.yaml") -> DataConfig:
    """Read ``configs/data.yaml`` and return a validated :class:`DataConfig`."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    profile = ProfileData.model_validate(raw)
    return profile.data


# Backwards-compatible alias
load_configs = load_compute_config
