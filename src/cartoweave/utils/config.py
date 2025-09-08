"""Utilities for dotted-path configuration access and validation."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Set


def get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Return ``cfg[path]`` where *path* uses dot notation.

    Missing keys return *default* without raising ``KeyError``.
    """
    cur: Any = cfg
    for part in path.split('.'):
        if not isinstance(cur, Mapping):
            return default
        if part not in cur:
            return default
        cur = cur[part]
    return cur


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten(key, v, out)
    else:
        out[prefix] = value


def flatten(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a flat ``dict`` mapping dotted paths to leaf values."""
    out: Dict[str, Any] = {}
    _flatten('', cfg, out)
    return out


def assert_known_keys(cfg: Mapping[str, Any], known: Iterable[str]) -> None:
    """Raise ``KeyError`` if *cfg* contains keys outside of *known*."""
    flat = flatten(cfg)
    unknown: Set[str] = set(flat) - set(known)
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
