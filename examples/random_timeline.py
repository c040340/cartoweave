"""Random timeline example honoring configuration layers."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from cartoweave.config.loader import load_configs, print_effective_config
from cartoweave.utils.dict_merge import deep_update
from cartoweave.data.random import get_scene
from cartoweave.api import solve_scene_script
from cartoweave.utils.numerics import is_finite_array

CACHE_PATH = os.environ.get("CARTOWEAVE_EXAMPLE_CACHE", "examples/_scene_cache.npz")
GENERATE_NEW = bool(int(os.environ.get("CARTOWEAVE_GENERATE_NEW", "1")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, help="WIDTHxHEIGHT override")
    parser.add_argument("--override", type=str, help="JSON blob overrides")
    parser.add_argument("--debug-nan", action="store_true", help="diagnose first NaN")
    args = parser.parse_args()

    cfg = load_configs(
        internals_path="../configs/solver.internals.yaml",
        tuning_path="../configs/solver.tuning.yaml",
        public_path="../configs/solver.public.yaml",
        viz_path="../configs/viz.yaml",
    )

    overrides: Dict[str, Any] = {}
    if args.frame:
        try:
            w, h = map(int, args.frame.lower().split("x"))
            overrides = deep_update(overrides, {"data": {"random": {"frame": {"width": w, "height": h}}}})
        except Exception:  # pragma: no cover - CLI parsing error
            pass
    if args.override:
        try:
            overrides = deep_update(overrides, json.loads(args.override))
        except Exception:  # pragma: no cover - bad JSON
            pass
    if overrides:
        def _flatten(d, prefix=""):
            for k, v in d.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    yield from _flatten(v, path)
                else:
                    yield path, v
        for path, val in _flatten(overrides):
            prev = cfg
            for key in path.split("."):
                if isinstance(prev, dict) and key in prev:
                    prev = prev[key]
                else:
                    prev = None
                    break
            print(f"[override] {path}: {prev} -> {val}")
        cfg = deep_update(cfg, overrides)

    print_effective_config(cfg)

    data_rand = cfg.get("data", {}).get("random", {})
    scene = get_scene(
        use_random=GENERATE_NEW,
        cache_path=CACHE_PATH,
        with_scene_script=True,
        gen_cfg=data_rand,
    )
    script = scene.get("scene_script") or {"steps": [{"name": "step0"}]}
    if isinstance(script, list):
        script = {"steps": script}
    plan = [{"name": "main"}]
    info = solve_scene_script(scene, script, cfg, solve_plan=plan)
    if args.debug_nan or os.environ.get("CARTOWEAVE_DEBUG_NAN") in {"1", "true", "True"}:
        hist = info.get("history", {})
        pos = hist.get("positions", [])
        recs = hist.get("records", [])
        nan_info = None
        for i, arr in enumerate(pos):
            if not is_finite_array(arr):
                phase = recs[i].get("meta", {}).get("stage_name") if i < len(recs) else None
                nan_info = {"frame": i, "field": "positions", "term": "total", "phase": phase}
                break
        if nan_info is None:
            for i, r in enumerate(recs):
                P_snap = r.get("P")
                if P_snap is not None and not is_finite_array(P_snap):
                    phase = r.get("meta", {}).get("stage_name")
                    nan_info = {"frame": i, "field": "records.P", "term": "total", "phase": phase}
                    break
                for k, v in (r.get("comps") or {}).items():
                    if not is_finite_array(v):
                        phase = r.get("meta", {}).get("stage_name")
                        nan_info = {"frame": i, "field": "records", "term": k, "phase": phase}
                        break
                if nan_info:
                    break
        if nan_info:
            print(
                f"FIRST_NAN frame={nan_info['frame']} field={nan_info['field']} term={nan_info['term']} phase={nan_info['phase']}"
            )
        else:
            print("FIRST_NAN none")
        print("ROOT_CAUSE_REPORT first_nan=%s" % (nan_info if nan_info else "none"))


if __name__ == "__main__":
    main()

