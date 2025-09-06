"""End-to-end smoke test constructing a SolvePack and running compute."""
from __future__ import annotations

import argparse
import json

import numpy as np

from cartoweave.compute import solve
from cartoweave.config.loader import load_configs
from cartoweave.data.api import build_solvepack_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Unused placeholder")
    ap.add_argument("--overrides", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    overrides = json.loads(args.overrides) if args.overrides else None
    merged = load_configs(overrides=overrides)
    merged["data"] = {"generate": {"num_labels": 5}}
    pack = build_solvepack_from_config(merged, seed=args.seed)
    pack.validate()
    view = solve(pack)
    last = view.last
    print(f"[e2e] N={pack.L} frames={len(view.frames)} final_E={last.E:.3g}")
    print(f"[e2e] P_norm={float(np.linalg.norm(last.P)):.3g}")


if __name__ == "__main__":
    main()
