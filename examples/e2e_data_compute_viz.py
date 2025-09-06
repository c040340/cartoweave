"""Minimal smoke test for data→compute without visualization."""

from __future__ import annotations

import argparse, json, yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve_behaviors
from cartoweave.compute.solver.solve_layer import StepReport


def _merge_override(cfg: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k in override:
        if k not in cfg:
            raise ValueError(f"unknown override key: {k}")
    merged = dict(cfg)
    merged.update(override)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--print-traj", action="store_true")
    ap.add_argument("--override", type=str, default=None)
    args = ap.parse_args()

    cfg_path = Path(__file__).resolve().parent.joinpath(args.config)
    with cfg_path.open("r", encoding="utf8") as f:
        cfg_data = yaml.safe_load(f)
    override = json.loads(args.override) if args.override else {}
    cfg_data = _merge_override(cfg_data, override)

    pack = build_solvepack_from_config(cfg_data, seed=args.seed)
    pack.validate()
    behaviors = pack.cfg.get("behaviors", [])
    if args.limit is not None:
        behaviors = behaviors[: args.limit]
        pack.cfg["behaviors"] = behaviors

    print(f"[e2e] N={pack.L} behaviors={len(behaviors)} seed={args.seed}")
    P_final, reps = solve_behaviors(pack, pack.cfg)
    if args.print_traj:
        for r in reps:
            print(r)
    last_inf = reps[-1].g_inf if reps else 0.0
    print(f"[e2e] done last_inf={last_inf:.6f} final_shape={P_final.shape}")


if __name__ == "__main__":
    main()
