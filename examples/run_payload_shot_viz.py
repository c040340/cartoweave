"""Run a specific shot from cartoweave_payload_latest.json and visualize.

Usage:
  python examples/run_payload_shot_viz.py \
      --file cartoweave_payload_latest.json \
      --shot 2 \
      [--margin 200] \
      [--viz]

Notes:
- This script avoids random scene generation and uses your provided payload.
- It computes a reasonable frame_size from element bbox + margin because
  solve_layout requires frame_size explicitly.
"""

from __future__ import annotations

import argparse
import json
from typing import Tuple

import numpy as np

from cartoweave import solve_layout


def bbox_frame(elements: dict, margin: float = 200.0) -> Tuple[float, float]:
    xs, ys = [], []
    for p in (elements.get("points") or []):
        xy = p.get("xy") if isinstance(p, dict) else p
        xs.append(float(xy[0])); ys.append(float(xy[1]))
    for ln in (elements.get("polylines") or []):
        arr = ln.get("polyline") if isinstance(ln, dict) else ln
        for x, y in arr:
            xs.append(float(x)); ys.append(float(y))
    for ar in (elements.get("polygons") or []):
        arr = ar.get("polygon") if isinstance(ar, dict) else ar
        for x, y in arr:
            xs.append(float(x)); ys.append(float(y))
    W = (max(xs) if xs else 0.0) + margin
    H = (max(ys) if ys else 0.0) + margin
    return max(W, 100.0), max(H, 100.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run and visualize a payload shot")
    ap.add_argument("--file", type=str, default="cartoweave_payload_latest.json", help="payload json path")
    ap.add_argument("--shot", type=int, default=2, help="0-based shot index (e.g., 2 for 3/14)")
    ap.add_argument("--margin", type=float, default=200.0, help="frame bbox margin (only used when --frame not provided)")
    ap.add_argument("--frame", nargs=2, type=float, default=None, help="explicit frame size: W H")
    ap.add_argument("--viz", action="store_true", help="open interactive viz window")
    args = ap.parse_args()

    payload = json.load(open(args.file, "r", encoding="utf-8"))
    calls = payload.get("calls") or []
    if not calls:
        raise SystemExit("payload has no calls[]")
    if args.shot < 0 or args.shot >= len(calls):
        raise SystemExit(f"shot index {args.shot} out of range 0..{len(calls)-1}")

    c = calls[args.shot]
    labels = c["labels"]
    elements = c["elements"]
    actions = c["actions"]
    cfg_profile = c.get("config_profile", "configs")

    if args.frame is not None:
        frame_size = (float(args.frame[0]), float(args.frame[1]))
    else:
        frame_size = bbox_frame(elements, margin=args.margin)

    print(f"Running shot={args.shot} with frame_size={frame_size} ...")
    res = solve_layout(
        labels=labels,
        elements=elements,
        actions=actions,
        frame_size=frame_size,
        config_profile=cfg_profile,
        return_viewpack=True,
    )

    vp = res.viewpack
    # print quick summary
    p = vp.passes[0] if vp.passes else None
    fr = vp.frames[p.t_end - 1] if p else vp.frames[-1]
    r = np.linalg.norm((fr.P - fr.anchors), axis=1)
    print("labels:", len(fr.P), "final r per label:", r.round(3))
    print("metrics:", {k: fr.metrics.get(k) for k in ("E", "gnorm", "g_inf")})

    if args.viz:
        from cartoweave.viz import show_vp
        print("Opening viz window ...")
        show_vp(vp, viz_cfg={"show": True, "panels": {"field": False}})


if __name__ == "__main__":  # pragma: no cover - example script
    main()
