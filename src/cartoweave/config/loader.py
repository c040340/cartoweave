"""Configuration loader for Cartoweave.

This module assembles configuration namespaces from YAML files, applies
optional overrides and profile presets, validates against the canonical schema
and returns a :class:`ConfigBundle` instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import logging
import yaml

from .schema import validate_config
from cartoweave.utils.dict_merge import deep_update
try:  # pragma: no cover - compatibility with schema alias
    from .schema import ConfigBundle
except ImportError:  # pragma: no cover
    from .schema import RootConfig as ConfigBundle

__all__ = ["load_configs", "print_effective_config"]

log = logging.getLogger(__name__)

# Presets expanding ``solver.public.profile`` into tuning overrides.
_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "basic": {},
    "fast": {"warmup": {"steps": 1}},
    "quality": {"warmup": {"steps": 5}},
}


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    """Return mapping loaded from ``path`` or ``{}`` if file is missing."""

    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Top-level YAML at {path} must be a mapping")
        return data


def load_configs(
    internals_path: str = "configs/solver.internals.yaml",
    tuning_path: str = "configs/solver.tuning.yaml",
    public_path: str = "configs/solver.public.yaml",
    viz_path: str = "configs/viz.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> ConfigBundle:
    """Load configuration files and return a :class:`ConfigBundle`."""

    internals_cfg = _read_yaml(internals_path)
    tuning_cfg = _read_yaml(tuning_path)
    public_cfg = _read_yaml(public_path)
    viz_cfg = _read_yaml(viz_path)

    if "internals" in internals_cfg and len(internals_cfg) == 1:
        internals_cfg = internals_cfg["internals"]
    if "tuning" in tuning_cfg and len(tuning_cfg) == 1:
        tuning_cfg = tuning_cfg["tuning"]
    if "public" in public_cfg and len(public_cfg) == 1:
        public_cfg = public_cfg["public"]

    data_cfg: Dict[str, Any] = {}
    data_cfg = deep_update(data_cfg, internals_cfg.pop("data", {}))
    data_cfg = deep_update(data_cfg, public_cfg.pop("data", {}))
    data_cfg = deep_update(data_cfg, tuning_cfg.pop("data", {}))

    log.info(f"[cfg] internals keys: {len(internals_cfg)}")
    log.info(f"[cfg] tuning    keys: {len(tuning_cfg)}")
    log.info(f"[cfg] public    keys: {len(public_cfg)}")
    log.info(f"[cfg] viz       keys: {len(viz_cfg)}")
    log.info(f"[cfg] data      keys: {len(data_cfg)}")

    root: Dict[str, Any] = {
        "solver": {
            "internals": internals_cfg,
            "tuning": tuning_cfg,
            "public": public_cfg,
        },
        "viz": viz_cfg,
        "data": data_cfg,
    }

    override_profile = (
        overrides.get("solver", {}).get("public", {}).get("profile") if overrides else None
    )
    profile = override_profile or public_cfg.get("profile")
    preset = _PROFILE_PRESETS.get(str(profile), {})
    if preset:
        root["solver"]["tuning"] = deep_update(root["solver"].get("tuning", {}), preset)

    if overrides:
        root = deep_update(root, overrides)

    validate_config(root)
    bundle = ConfigBundle(**root)
    return bundle


def print_effective_config(bundle: ConfigBundle, path: str | None = None) -> None:
    """Print or dump a fully-populated configuration bundle."""

    snap = bundle.model_dump(
        exclude_unset=False, exclude_defaults=False, exclude_none=True
    )
    text = yaml.safe_dump(snap, sort_keys=False, allow_unicode=True)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print("Effective Config:\n" + text)

