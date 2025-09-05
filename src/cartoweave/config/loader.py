"""Load Cartoweave configuration files in a fixed order.

The loader merges YAML files in the following order:
    1. ``solver.internals.yaml`` – numeric safety and hard limits
    2. ``solver.tuning.yaml``    – tuning knobs
    3. ``solver.public.yaml``    – product level switches
    4. ``viz.yaml``               – visualization options (separate namespace)
    5. in-memory overrides provided to :func:`load_configs`

Deprecated keys listed in ``deprecations.yaml`` are migrated to their new
locations and an orphan check ensures no unknown keys slipped in.  The selected
``solver.public.profile`` may expand into additional tuning keys before
validation.  Visualization keys never override solver values.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Iterable

import warnings
import yaml

from .schema import validate_config

__all__ = ["load_configs", "print_effective_config", "migrate_deprecated_keys", "validate_no_orphans"]

_EFFECTIVE_CONFIG: Dict[str, Dict[str, Any]] | None = None

# presets expanding ``solver.public.profile`` into tuning overrides
_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "basic": {},
    "fast": {"warmup": {"steps": 1}},
    "quality": {"warmup": {"steps": 5}},
}


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Top-level YAML at {path} must be a mapping")
        return data


def _deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(dict(base[k]), v)
        else:
            base[k] = deepcopy(v)
    return base


def _iter_paths(d: Dict[str, Any], prefix: str = "") -> Iterable[str]:
    for k, v in d.items():
        new_prefix = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            yield from _iter_paths(v, new_prefix)
        else:
            yield new_prefix


def _pop_by_path(d: Dict[str, Any], parts: Iterable[str]) -> Any:
    cur = d
    parts = list(parts)
    stack = []
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return None
        stack.append((cur, p))
        cur = cur[p]
    if isinstance(cur, dict) and parts[-1] in cur:
        val = cur.pop(parts[-1])
        while stack and not cur:
            cur_parent, key = stack.pop()
            del cur_parent[key]
            cur = cur_parent
        return val
    return None


def _set_by_path(d: Dict[str, Any], parts: Iterable[str], value: Any) -> None:
    cur = d
    parts = list(parts)
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def migrate_deprecated_keys(cfg: Dict[str, Any], mappings: Dict[str, str]) -> None:
    for old, new in mappings.items():
        val = _pop_by_path(cfg, old.split("."))
        if val is not None:
            _set_by_path(cfg, new.split("."), val)
            warnings.warn(
                f"Config key '{old}' is deprecated; use '{new}'", UserWarning,
                stacklevel=2,
            )


def validate_no_orphans(
    cfg: Dict[str, Any], known_schema: set[str], mappings: Dict[str, str]
) -> None:
    allowed = set(known_schema) | set(mappings.keys())
    unknown = [p for p in _iter_paths(cfg) if p not in allowed]
    if unknown:
        raise KeyError(f"Unknown config keys: {', '.join(sorted(unknown))}")


def load_configs(
    internals_path: str = "configs/solver.internals.yaml",
    tuning_path: str = "configs/solver.tuning.yaml",
    public_path: str = "configs/solver.public.yaml",
    viz_path: str = "configs/viz.yaml",
    deprecations_path: str = "configs/deprecations.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load configuration files and return ``{"solver": ..., "viz": ...}``.

    The ``solver`` dictionary is built by deep-merging ``internals``, ``tuning``
    and ``public`` in that order.  ``viz`` settings are loaded separately so they
    never override solver keys.  Deprecated keys are migrated according to
    ``deprecations.yaml`` and an orphan check ensures no unknown keys remain.
    """
    solver_cfg: Dict[str, Any] = {}
    for path in (internals_path, tuning_path, public_path):
        solver_cfg = _deep_update(solver_cfg, _read_yaml(path))

    viz_cfg: Dict[str, Any] = _read_yaml(viz_path)

    root: Dict[str, Any] = {"solver": solver_cfg, "viz": viz_cfg}
    mappings: Dict[str, str] = _read_yaml(deprecations_path)

    override_profile = (
        overrides.get("solver", {}).get("public", {}).get("profile")
        if overrides
        else None
    )
    profile = override_profile or root.get("solver", {}).get("public", {}).get("profile")
    preset = _PROFILE_PRESETS.get(str(profile), {})
    if preset:
        root["solver"]["tuning"] = _deep_update(root["solver"].get("tuning", {}), preset)

    known_schema = set(_iter_paths(root)) | set(mappings.values())

    if overrides:
        root = _deep_update(root, overrides)

    validate_no_orphans(root, known_schema, mappings)
    migrate_deprecated_keys(root, mappings)
    validate_no_orphans(root, known_schema, {})
    validate_config(root)

    global _EFFECTIVE_CONFIG
    _EFFECTIVE_CONFIG = {"solver": deepcopy(root["solver"]), "viz": deepcopy(root["viz"])}

    return _EFFECTIVE_CONFIG


def print_effective_config(path: str | Path | None = None) -> None:
    """Dump the most recently loaded configuration.

    If ``path`` is provided, the snapshot is written to that file as YAML.
    Otherwise it is printed to stdout with a header ``"Effective Config"``.
    """
    if _EFFECTIVE_CONFIG is None:
        raise RuntimeError("load_configs has not been called yet")
    data = yaml.safe_dump(_EFFECTIVE_CONFIG, sort_keys=True)
    if path is None:
        print("Effective Config:\n" + data)
    else:
        p = Path(path)
        p.write_text(data, encoding="utf-8")

