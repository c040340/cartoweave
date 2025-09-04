from __future__ import annotations
from typing import Dict, Any, Literal, Optional
from .schema import SPEC, spec_of, Mutability

Phase = Literal["load","action_begin","runtime"]


class ConfigValidationError(ValueError): ...
class ConfigMutabilityError(ValueError): ...
class ConfigTypeRangeError(ValueError): ...


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
            if cur == prev:
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
