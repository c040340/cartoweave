"""Minimal e2e script building a pack and exiting before solving."""

from __future__ import annotations

import argparse, json

from cartoweave.config.loader import load_configs
from cartoweave.data.api import build_solvepack_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Profile YAML path")
    ap.add_argument("--overrides", type=str, default=None, help="JSON overrides")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    overrides = json.loads(args.overrides) if args.overrides else None
    merged = load_configs(profile=args.config, overrides=overrides)

    pack = build_solvepack_from_config(merged, seed=args.seed)
    pack.validate()
    scene = pack.scene0
    print(
        f"[e2e] N={pack.N} steps={len(pack.cfg.get('behaviors', []))} seed={args.seed}"
    )
    print(
        "[e2e] scene points/lines/areas = "
        f"{len(scene.points)}/{len(scene.lines)}/{len(scene.areas)}"
    )
    print("[e2e] pack ready (pre-solve)")


if __name__ == "__main__":
    main()
