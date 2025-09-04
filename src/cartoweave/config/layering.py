from __future__ import annotations

import numpy as np

from typing import Dict, Any, Literal, Optional
from .schema import SPEC, spec_of, Mutability
from .profiles.shapes import SHAPE_PROFILES, SIGMA_SCALE_KEYS
from .profiles.calibration import CALIB_PROFILES

Phase = Literal["load","action_begin","runtime"]


class ConfigValidationError(ValueError): ...
class ConfigMutabilityError(ValueError): ...
class ConfigTypeRangeError(ValueError): ...


def _values_equal(a, b) -> bool:
    """Safe equality that works for numpy arrays and scalars."""
    # 同一个对象
    if a is b:
        return True
    # numpy array
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(a, b)
        except Exception:
            return False
    # 其他类型：尽量用 ==
    try:
        return a == b
    except Exception:
        return False


def _type_ok(v: Any, t: type) -> bool:
    if t is float and isinstance(v, (int, float)):
        return True
    return isinstance(v, t)


def _in_range(v: Any, lo_hi: Optional[tuple]) -> bool:
    if lo_hi is None:
        return True
    lo, hi = lo_hi
    return (v >= lo) and (v <= hi)


def _in_choices(v: Any, choices: Optional[tuple]) -> bool:
    return True if choices is None else (v in choices)


def _check_value(key: str, val: Any) -> None:
    si = spec_of(key)
    if si is None:
        return
    if "type" in si and not _type_ok(val, si["type"]):
        raise ConfigTypeRangeError(f"[type] {key}: {type(val).__name__} != {si['type'].__name__}")
    if "range" in si and si["type"] in (int, float) and not _in_range(float(val), si["range"]):
        lo, hi = si["range"]  # type: ignore
        raise ConfigTypeRangeError(f"[range] {key}: {val} not in [{lo}, {hi}]")
    if "choices" in si and not _in_choices(val, si["choices"]):
        raise ConfigTypeRangeError(f"[choices] {key}: {val} not in {si['choices']}")


def validate_cfg(cfg: Dict[str, Any], phase: Phase) -> None:
    """Validate types/ranges universally; mutability rules by phase (no side effects)."""
    if phase not in ("load","action_begin","runtime"):
        raise ConfigValidationError(f"Unknown phase: {phase}")

    # 1) type/range for all present keys
    for k, v in cfg.items():
        _check_value(k, v)

    # 2) mutability checks require reference snapshots; caller passes them in cfg if available:
    #    - "_snapshot_load": baseline after load/merge
    #    - "_snapshot_action": baseline at action_begin
    ref_key = "_snapshot_load" if phase in ("action_begin","runtime") else None
    if phase == "runtime":
        ref_key = "_snapshot_action"

    if ref_key and ref_key in cfg and isinstance(cfg[ref_key], dict):
        ref: Dict[str, Any] = cfg[ref_key]
        for k, cur in cfg.items():
            if k.startswith("_"):
                continue
            if k not in ref:
                # New keys are allowed; enforce mutability if spec exists
                si = spec_of(k)
                if si and si.get("mutable") == "frozen" and phase != "load":
                    raise ConfigMutabilityError(f"[frozen/new] {k} cannot be introduced in phase={phase}")
                continue
            prev = ref[k]
            if _values_equal(cur, prev):
                continue
            si = spec_of(k)
            if not si:
                # unknown keys allowed to change
                continue
            mut: Mutability = si.get("mutable", "action")
            if phase == "action_begin" and mut == "frozen":
                raise ConfigMutabilityError(f"[frozen] {k} cannot change at action_begin (prev={prev}, cur={cur})")
            if phase == "runtime" and mut != "runtime":
                raise ConfigMutabilityError(f"[immutable] {k} cannot change at runtime (prev={prev}, cur={cur})")


def snapshot(cfg: Dict[str, Any], name: str) -> None:
    """Store a shallow snapshot inside cfg for later diff checks."""
    if name not in ("_snapshot_load","_snapshot_action"):
        raise ValueError("snapshot name must be _snapshot_load or _snapshot_action")
    cfg[name] = {k: v for k, v in cfg.items() if not k.startswith("_")}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new dict = a ⊕ b (b has precedence). Does not mutate inputs.
    Nested dicts are merged recursively.
    """
    out: Dict[str, Any] = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_base_cfg() -> Dict[str, Any]:
    """
    Load and merge minimal presets: base + scene_defaults.
    Validate at 'load' phase and snapshot baseline for later diffs.
    """
    from .presets.base import base_defaults
    from .presets.scene_defaults import scene_defaults

    cfg = _deep_merge(base_defaults(), scene_defaults())
    # Type/range validation only; unknown keys are allowed.
    validate_cfg(cfg, phase="load")
    # Store the post-load snapshot (baseline for future mutability checks).
    snapshot(cfg, "_snapshot_load")
    return cfg


def _resolve_shape_profile(name: str) -> Dict[str, Any]:
    """Return a merged profile dict with inheritance resolved (child overrides base)."""
    if name not in SHAPE_PROFILES:
        raise KeyError(f"Unknown shape profile: {name}")
    prof = SHAPE_PROFILES[name]
    fixed: Dict[str, Any] = dict(prof.get("fixed", {}))
    parent = prof.get("extends")
    if parent:
        base = _resolve_shape_profile(parent)
        merged = dict(base.get("fixed", {}))
        merged.update(fixed)
        return {"fixed": merged}
    return {"fixed": fixed}


def apply_shape_profile(cfg: Dict[str, Any],
                        name: Optional[str],
                        enable: bool = False,
                        sigma_scale: float = 1.0) -> int:
    """
    Apply a frozen-only shape profile onto cfg.
    Only overrides keys that already exist in cfg and are frozen in SPEC.
    Returns the number of keys patched.
    """
    if not enable or not name:
        return 0
    resolved = _resolve_shape_profile(str(name))
    fixed: Dict[str, Any] = resolved.get("fixed", {})
    patched = 0
    for k, v in fixed.items():
        if k not in cfg:
            continue
        si = spec_of(k)
        if not si or si.get("mutable") != "frozen":
            continue
        val = float(v)
        if k in SIGMA_SCALE_KEYS:
            try:
                val = float(v) * float(sigma_scale)
            except Exception:
                val = float(v)
        cfg[k] = val
        patched += 1
    return patched


def apply_calib_profile(cfg: Dict[str, Any], name: str, fill_only: bool = True) -> int:
    """Merge a calibration profile into ``cfg``.

    Parameters
    ----------
    cfg:
        Configuration dictionary to be patched in-place.
    name:
        Name of the profile defined in ``CALIB_PROFILES``.
    fill_only:
        When ``True`` (default), only keys that are missing are written. Existing
        values are preserved. When ``False``, profile values override existing
        ones.

    Returns
    -------
    int
        Number of individual keys filled/overridden.
    """
    if not name:
        raise KeyError("Calibration profile name is required")
    prof = CALIB_PROFILES.get(name)
    if prof is None:
        raise KeyError(f"Unknown calibration profile: {name}")

    patched = 0

    def _patch_container(container_key: str, items: Dict[str, Any]) -> None:
        nonlocal patched
        container = cfg.get(container_key)
        if not isinstance(container, dict):
            if fill_only and container_key in cfg and not isinstance(cfg[container_key], dict):
                return
            container = {}
            cfg[container_key] = container
        for k, v in items.items():
            if fill_only and k in container:
                continue
            container[k] = v
            patched += 1

    # target ratios
    if "target_rel" in prof:
        _patch_container("calib.k.target_rel", prof["target_rel"])

    # trigger thresholds
    if "trigger" in prof:
        _patch_container("calib.trigger", prof["trigger"])

    # quantiles and clamps share the same container
    if "quantiles" in prof:
        _patch_container("calib.k", prof["quantiles"])
    if "limits" in prof:
        _patch_container("calib.k", prof["limits"])

    # scalar EMA parameter
    if "ema_alpha" in prof:
        key = "calib.k.ema_alpha"
        if not (fill_only and key in cfg):
            cfg[key] = float(prof["ema_alpha"])
            patched += 1

    return patched
