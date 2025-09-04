from __future__ import annotations
from typing import Dict, Any
from cartoweave.config.layering import load_base_cfg, validate_cfg

def test_load_base_cfg_merges_and_validates():
    cfg = load_base_cfg()
    # Basic presence checks
    assert isinstance(cfg, dict)
    assert cfg.get("logging.level") in ("DEBUG","INFO","WARNING","ERROR")
    assert "scene.width_px" in cfg and "scene.height_px" in cfg

    # Validate again (idempotent); should not raise
    validate_cfg(cfg, phase="load")

def test_deep_merge_non_mutating():
    from cartoweave.config.layering import _deep_merge
    a = {"x": {"y": 1}, "k": 3}
    b = {"x": {"z": 2}, "k": 4}
    c = _deep_merge(a, b)
    assert c == {"x": {"y": 1, "z": 2}, "k": 4}
    # originals unchanged
    assert a == {"x": {"y": 1}, "k": 3}
    assert b == {"x": {"z": 2}, "k": 4}
