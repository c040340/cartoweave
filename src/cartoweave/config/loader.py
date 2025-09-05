"""Configuration loader for Cartoweave.

This module assembles configuration namespaces from YAML files, applies
optional overrides and profile presets, validates against the canonical schema
and returns a merged configuration mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import logging
import yaml

from .schema import validate_config
from cartoweave.utils.dict_merge import deep_update

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


def _inject_legacy_term_keys(cfg: Dict[str, Any]) -> None:
    """Populate legacy dotted keys expected by older force implementations."""
    terms = cfg.get("terms", {})
    anc = terms.get("anchor", {}).get("spring", {})
    if anc:
        cfg["anchor.k.spring"] = anc.get("k")
        r0 = anc.get("r0")
        if r0 is not None:
            cfg["anchor.r0.points"] = r0
            cfg["anchor.r0.lines"] = r0
            cfg["anchor.r0.areas"] = r0
    foc = terms.get("focus", {})
    if foc:
        cfg["focus.k.attract"] = foc.get("k")
        sig = foc.get("sigma")
        cfg["focus.sigma.x"] = sig
        cfg["focus.sigma.y"] = sig
    bnd = terms.get("boundary", {})
    if bnd:
        cfg["boundary.k.wall"] = bnd.get("k")
    rep = terms.get("label_label_repulse", {})
    if rep:
        cfg["ll.k.repulse"] = rep.get("k")
    ins = terms.get("label_label_inside", {})
    if ins:
        cfg["ll.k.inside"] = ins.get("k")
    rep = terms.get("line_label_repulse", {})
    if rep:
        cfg["ln.k.repulse"] = rep.get("k")
    ins = terms.get("line_label_inside", {})
    if ins:
        cfg["ln.k.inside"] = ins.get("k")
    rep = terms.get("point_label_repulse", {})
    if rep:
        cfg["pl.k.repulse"] = rep.get("k")
    ins = terms.get("point_label_inside", {})
    if ins:
        cfg["pl.k.inside"] = ins.get("k")
    ac = terms.get("area_cross", {})
    if ac:
        cfg["area.k.cross"] = ac.get("k")
    ae = terms.get("area_embed", {})
    if ae:
        cfg["area.k.embed"] = ae.get("k")


def load_configs(
    internals_path: str = "configs/solver.internals.yaml",
    tuning_path: str = "configs/solver.tuning.yaml",
    public_path: str = "configs/solver.public.yaml",
    viz_path: str = "configs/viz.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load configuration files and return a merged mapping."""

    internals_cfg = _read_yaml(internals_path)
    tuning_cfg = _read_yaml(tuning_path)
    public_cfg = _read_yaml(public_path)
    viz_cfg = _read_yaml(viz_path)

    surf = viz_cfg.get("surface3d", {})
    if isinstance(surf, dict) and surf.get("enable"):
        viz_cfg.setdefault("panels", {})["field"] = True
        viz_cfg.setdefault("field", {})["mode"] = "surface3d"

    if "internals" in internals_cfg and len(internals_cfg) == 1:
        internals_cfg = internals_cfg["internals"]
    if "tuning" in tuning_cfg and len(tuning_cfg) == 1:
        tuning_cfg = tuning_cfg["tuning"]
    if "public" in public_cfg and len(public_cfg) == 1:
        public_cfg = public_cfg["public"]

    cfg: Dict[str, Any] = {}
    cfg = deep_update(cfg, internals_cfg)
    cfg = deep_update(cfg, public_cfg)
    cfg = deep_update(cfg, tuning_cfg)
    cfg = deep_update(cfg, {"viz": viz_cfg})

    override_profile = (
        overrides.get("solver", {}).get("profile") if overrides else None
    )
    profile = override_profile or cfg.get("solver", {}).get("profile")
    preset = _PROFILE_PRESETS.get(str(profile), {})
    if preset:
        cfg["solver"] = deep_update(cfg.get("solver", {}), preset)

    if overrides:
        cfg = deep_update(cfg, overrides)

    validate_config(cfg)
    _inject_legacy_term_keys(cfg)
    return cfg


def print_effective_config(cfg: Dict[str, Any], path: str | None = None) -> None:
    """Print or dump a fully-populated configuration."""

    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print("Effective Config:\n" + text)

