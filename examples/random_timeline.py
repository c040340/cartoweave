"""Random scene + short timeline (config-compatible, T1–T7).

- Generates or loads a cached random scene.
- Loads layered config and fills calibration strategy (fill-only).
- Keeps calibration gates OFF for stable example output.
- Runs a short 2-step timeline using the public API (orchestrator).
"""

from __future__ import annotations
from typing import Dict, Any
import os
import numpy as np

from cartoweave.config.layering import load_base_cfg, apply_calib_profile
from cartoweave.data.random import generate_scene, save_scene, load_scene
from cartoweave.orchestrators.timeline import run_timeline
from cartoweave.engine.solvers.hybrid import solve_layout_hybrid as hybrid_solve

_CACHE = os.environ.get("CARTOWEAVE_EXAMPLE_CACHE", "examples/_scene_cache.npz")

def _get_scene() -> Dict[str, Any]:
    if os.path.exists(_CACHE):
        return load_scene(_CACHE)
    data = generate_scene(canvas_size=(1920, 1080),
                          n_points=12, n_lines=2, n_areas=1, seed=42)
    save_scene(data, _CACHE)
    return data

def _build_cfg() -> Dict[str, Any]:
    cfg = load_base_cfg()
    apply_calib_profile(cfg, cfg.get("calib.k.profile", "default"), fill_only=True)

    # Keep gates OFF in examples:
    cfg["calib.shape.enable"] = False
    cfg["calib.k.enable"]     = False

    # Viewer flags (examples should not pop UI by default)
    cfg["viz.show"] = True
    cfg["viz.field.kind"] = cfg.get("viz.field.kind", "heatmap")
    cfg["viz.field.cmap"] = cfg.get("viz.field.cmap", "viridis")
    return cfg

def _build_schedule():
    # A short, generic two-step schedule; actual implementation lives in orchestrator
    return [
        {"name": "warmup_no_anchor", "scale": {"anchor.k.spring": 0.0}},
        {"name": "main_solve"},
    ]

def main():
    scene = _get_scene()
    cfg   = _build_cfg()
    schedule = _build_schedule()

    results = run_timeline(schedule, cfg, hybrid_solve)  # 返回 [(P_out, info), ...]
    last_P, last_info = results[-1]
    print("[random_timeline] steps:", len(results), "last labels:", len(scene.get("labels", [])))

if __name__ == "__main__":
    main()
