# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys, pathlib
import numpy as np

from .contracts.solvepack import SolvePack
from .compute.solve import solve
from .logging import init_logging_from_cfg


def _load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(p: str, obj):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def cmd_solve(args):
    cfg = _load_json(args.config) if args.config.endswith(".json") else _load_json(args.config)
    init_logging_from_cfg(cfg)
    scene = _load_json(args.scene)

    L = int(args.L) if args.L else len(scene.get("labels_init", [])) or 0
    if L <= 0:
        raise SystemExit("Cannot infer L; pass --L or provide scene['labels_init'].")

    P0 = np.zeros((L, 2), dtype=float)
    if args.P0 and pathlib.Path(args.P0).exists():
        P0 = np.asarray(_load_json(args.P0), dtype=float)
        if P0.shape != (L, 2):
            raise SystemExit(f"P0 shape {P0.shape} != {(L,2)}")
    active = np.ones(L, dtype=bool)

    sp = SolvePack(
        L=L,
        P0=P0,
        active0=active,
        labels0=[],
        scene0=scene,
        cfg=cfg,
        actions=[],
        action_num=None,
        behaviors=[],
    )
    view = solve(sp)

    out_sum = args.out_summary or "out/summary.json"
    out_P = args.out_P or "out/P.json"
    _dump_json(out_sum, view.summary)
    _dump_json(out_P, view.last.P.tolist())
    if args.print:
        print(json.dumps({"summary": view.summary, "P_last": view.last.P.tolist()}, ensure_ascii=False))
    return 0


def make_parser():
    p = argparse.ArgumentParser(prog="cartoweave")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("solve", help="Run a solve from config + scene")
    ps.add_argument("--config", required=True, help="compute-only config JSON")
    ps.add_argument("--scene", required=True, help="scene JSON")
    ps.add_argument("--L", type=int, default=None, help="number of labels if not in scene")
    ps.add_argument("--iters", type=int, default=10, help="fallback iters if not in cfg")
    ps.add_argument("--P0", default=None, help="optional JSON array of shape (L,2)")
    ps.add_argument("--out-summary", dest="out_summary", default="out/summary.json")
    ps.add_argument("--out-P", dest="out_P", default="out/P.json")
    ps.add_argument("--print", action="store_true", help="print JSON result to stdout")
    ps.set_defaults(func=cmd_solve)

    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = make_parser()
    ns = parser.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
