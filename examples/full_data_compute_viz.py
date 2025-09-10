"""End-to-end smoke test constructing a SolvePack and running compute."""
from __future__ import annotations

import argparse
import json

import numpy as np

from cartoweave.compute import solve
from cartoweave.config.loader import load_configs
from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.viz import show_vp
from cartoweave.logging import init_logging_from_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    overrides = json.loads(args.overrides) if args.overrides else None
    compute_cfg = load_configs(overrides=overrides)["compute"]
    pack = make_solvepack_from_data_defaults(
        compute_cfg=compute_cfg, data_path="configs/data.yaml"
    )
    pack.validate()
    init_logging_from_cfg(getattr(pack, "cfg", None))
    view = solve(pack)
    last = view.frames[-1]
    print(f"[e2e] N={pack.L} frames={len(view.frames)} final_E={last.E:.3g}")
    print(f"[e2e] P_norm={float(np.linalg.norm(last.P)):.3g}")

    show_vp(view)


if __name__ == "__main__":
    main()
