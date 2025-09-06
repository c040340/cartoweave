"""Minimal smoke test for data→compute without visualization."""

from __future__ import annotations

import argparse, json
from typing import Dict, Any, List

from cartoweave.config.loader import load_configs
from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve_behaviors
from cartoweave.compute.solver.solve_layer import StepReport


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Profile YAML path")
    ap.add_argument("--overrides", type=str, default=None, help="JSON overrides")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--print-traj", action="store_true")
    args = ap.parse_args()

    overrides = json.loads(args.overrides) if args.overrides else None
    merged = load_configs(profile=args.config, overrides=overrides)

    pack = build_solvepack_from_config(merged, seed=args.seed)
    pack.validate()
    behaviors = pack.cfg.get("behaviors", [])
    if args.limit is not None:
        behaviors = behaviors[: args.limit]
        pack.cfg["behaviors"] = behaviors

    print(f"[e2e] N={pack.N} steps={len(behaviors)} seed={args.seed}")
    P_final, reps = solve_behaviors(pack, pack.cfg)
    if args.print_traj:
        for r in reps:
            print(r)
    print(f"[e2e] done final_shape={pack.P0.shape}")


if __name__ == "__main__":
    main()
