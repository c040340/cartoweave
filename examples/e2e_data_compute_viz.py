"""Minimal end-to-end example using SolveLayer without visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.compute.solver.solve_layer import (
    run_iters,
    SolveContext,
    StepReport,
)


def _energy_no_meta(P, labels, scene, active, cfg):
    """Adapter stripping meta return value from energy_and_grad_full."""
    E, G, comps, _ = energy_and_grad_full(P, labels, scene, active, cfg)
    return E, G, comps


def solve_behaviors(pack, cfg) -> Tuple[np.ndarray, List[StepReport]]:
    """Iterate SolveLayer over all configured behaviors/stages."""
    P = pack.P0.copy()
    behaviors = getattr(pack, "behaviors", getattr(pack, "stages", []))
    labels = pack.scene.get("labels")
    reps: List[StepReport] = []
    for beh in behaviors:
        active = beh.get("mask_override", pack.active_mask0)
        ctx = SolveContext(
            labels=labels,
            scene=pack.scene,
            active=active,
            cfg=cfg,
            iters=int(beh.get("iters", 0)),
            mode=str(beh.get("solver", "lbfgs")),
            params=beh.get("params", {}),
        )
        P, reps = run_iters(P, ctx, _energy_no_meta, report=True)
    return P, reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)  # unused placeholder
    args = ap.parse_args()

    cfg_path = Path(__file__).resolve().parent.joinpath(args.config)
    pack = build_solvepack_from_config(str(cfg_path), seed=args.seed)
    behaviors = getattr(pack, "behaviors", getattr(pack, "stages", []))
    print(f"[e2e] N={pack.L} behaviors={len(behaviors)}")

    P_final, reps = solve_behaviors(pack, pack.cfg)
    last_inf = reps[-1].g_inf if reps else 0.0
    print(f"[e2e] done last_inf={last_inf:.6f} final_shape={P_final.shape}")


if __name__ == "__main__":
    main()

