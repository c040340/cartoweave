#!/usr/bin/env python3
"""Diagnose anchor vs. label-center overlap for a single-point case.

This script builds a minimal scene from:
  - anchor_xy (point element)
  - label WH (rect label anchored to that point)
and runs solve_layout under multiple force configurations to isolate causes.

It prints, for each run:
  - final position, delta to anchor, distance r
  - per-term force vectors (at final frame)
  - energy metrics (E, gnorm, g_inf)
  - s_hit estimate (r - rho) for the anchor spring model

Optionally, it opens the interactive visualization.

Usage:
  python scripts/diagnostics/anchor_debug.py \
      --anchor 425.3176 707.5773 \
      --wh 717.68 104.0 \
      --frame 1600 1600 \
      --config configs \
      [--viz]

"""

from __future__ import annotations

import argparse
from typing import Dict, Any, Sequence

import numpy as np

from cartoweave import solve_layout


def _bbox_based_frame(anchor: Sequence[float], wh: Sequence[float], margin: float = 200.0):
    ax, ay = float(anchor[0]), float(anchor[1])
    hx, hy = float(wh[0]) * 0.5, float(wh[1]) * 0.5
    W = ax + hx + float(margin)
    H = ay + hy + float(margin)
    return (max(W, 100.0), max(H, 100.0))


def _solve_once(anchor_xy, wh, frame_size, config_profile, overrides=None, return_viewpack=False):
    labels = [
        {
            "label_id": 0,
            "WH": [float(wh[0]), float(wh[1])],
            "anchor": {"kind": "point", "element_id": 0},
            "mode": "rect",
        }
    ]
    elements = {"points": [{"element_id": 0, "xy": [float(anchor_xy[0]), float(anchor_xy[1])]}]}
    actions = [{"t": 0, "op": "activate", "label_ids": [0]}]
    res = solve_layout(
        labels=labels,
        elements=elements,
        actions=actions,
        frame_size=frame_size,
        config_profile=config_profile,
        overrides=overrides,
        return_viewpack=return_viewpack,
        deterministic_seed=42,
    )
    return res


def _estimate_s_hit(P: np.ndarray, anchor: np.ndarray, wh: np.ndarray, eps_abs: float = 1e-6) -> float:
    """Approximate s_hit = r - rho using the same smooth-norm model as anchor.spring.

    This uses the r>eps branch (typical for the final position)."""
    cx, cy = float(P[0]), float(P[1])
    ax, ay = float(anchor[0]), float(anchor[1])
    ux, uy = cx - ax, cy - ay
    r = float((ux * ux + uy * uy) ** 0.5)
    if r <= 1e-12:
        # fall back to diagonal direction
        uxh, uyh = 0.70710678, 0.70710678
        rho = 0.5 * float(wh[0]) * abs(uxh) + 0.5 * float(wh[1]) * abs(uyh)
        return r - rho
    hx, hy = 0.5 * float(wh[0]), 0.5 * float(wh[1])
    eps2 = float(eps_abs) * float(eps_abs)
    A = (ux * ux + eps2 * r * r) ** 0.5
    B = (uy * uy + eps2 * r * r) ** 0.5
    C = hx * A + hy * B
    rho = float(C / max(r, 1e-12))
    return r - rho


def _print_run(title: str, vp, anchor_xy):
    frames = vp.frames
    last = frames[vp.passes[0].t_end - 1] if vp.passes else frames[-1]
    P = np.asarray(last.P, float)[0]
    comps: Dict[str, np.ndarray] = last.comps
    E = float(last.metrics.get("E", 0.0))
    gnorm = float(last.metrics.get("gnorm", 0.0))
    g_inf = float(last.metrics.get("g_inf", 0.0))
    anchor = np.asarray(anchor_xy, float)
    wh = np.asarray(vp.WH[0] if vp.WH is not None else [0.0, 0.0], float)
    delta = P - anchor
    r = float(np.linalg.norm(delta))
    s_hit = _estimate_s_hit(P, anchor, wh)

    print(f"\n=== {title} ===")
    print(f"final P: {P}, anchor: {anchor}")
    print(f"delta: {delta}, r: {r:.6f}, s_hit (r-rho): {s_hit:.6f}")
    print(f"E: {E:.6f}, gnorm: {gnorm:.6f}, g_inf: {g_inf:.6f}")
    # Optional step-limit metrics if present
    sl_scale = last.metrics.get("step_limit_scale")
    if sl_scale is not None:
        print(f"step_limit_scale: {float(sl_scale):.6f}")
    if last.meta and isinstance(last.meta, dict):
        evs = last.meta.get("events", [])
        if evs:
            # print just the last optimizer_step, if any
            opt = [e for e in evs if e.get("kind") == "optimizer_step"]
            if opt:
                o = opt[-1]
                print("optimizer_step:", {k: o.get(k) for k in ["algo", "iter_in_algo", "step_size", "ls_evals", "wolfe", "delta_E", "gnorm"]})
    terms = sorted(comps.keys())
    print("terms:", terms)
    for k in terms:
        v = np.asarray(comps[k], float)[0]
        print(f"  {k:14s}: F = {v}")


def main():
    ap = argparse.ArgumentParser(description="Diagnose anchor vs. label-center overlap")
    ap.add_argument("--anchor", nargs=2, type=float, required=True, help="anchor xy")
    ap.add_argument("--wh", nargs=2, type=float, required=True, help="label WH")
    ap.add_argument("--frame", nargs=2, type=float, default=None, help="frame size WxH")
    ap.add_argument("--config", type=str, default="configs", help="config profile path or name")
    ap.add_argument("--viz", action="store_true", help="enable interactive viz")
    args = ap.parse_args()

    anchor_xy = [args.anchor[0], args.anchor[1]]
    wh = [args.wh[0], args.wh[1]]
    frame_size = tuple(args.frame) if args.frame else _bbox_based_frame(anchor_xy, wh)

    # Baseline run (all defaults)
    res = _solve_once(anchor_xy, wh, frame_size, args.config, return_viewpack=True)
    _print_run("baseline (all forces)", res.viewpack, anchor_xy)

    # Anchor only
    o_anchor_only = {
        "compute": {
            "public": {
                "forces": {
                    "pl": {"rect": {"enable": False}},
                    "ll": {"rect": {"enable": False}, "disk": {"enable": False}},
                    "area": {"cross": {"enable": False}, "embed": {"enable": False}, "softout": {"enable": False}},
                    "boundary": {"wall": {"enable": False}},
                    "focus": {"attract": {"enable": False}},
                    "anchor": {"spring": {"enable": True}},
                }
            }
        }
    }
    res2 = _solve_once(anchor_xy, wh, frame_size, args.config, overrides=o_anchor_only, return_viewpack=True)
    _print_run("anchor.spring only", res2.viewpack, anchor_xy)

    # Anchor + boundary
    o_ab = {
        "compute": {
            "public": {
                "forces": {
                    "pl": {"rect": {"enable": False}},
                    "ll": {"rect": {"enable": False}, "disk": {"enable": False}},
                    "area": {"cross": {"enable": False}, "embed": {"enable": False}, "softout": {"enable": False}},
                    "boundary": {"wall": {"enable": True}},
                    "focus": {"attract": {"enable": False}},
                    "anchor": {"spring": {"enable": True}},
                }
            }
        }
    }
    res3 = _solve_once(anchor_xy, wh, frame_size, args.config, overrides=o_ab, return_viewpack=True)
    _print_run("anchor.spring + boundary.wall", res3.viewpack, anchor_xy)

    # Anchor + focus
    o_af = {
        "compute": {
            "public": {
                "forces": {
                    "pl": {"rect": {"enable": False}},
                    "ll": {"rect": {"enable": False}, "disk": {"enable": False}},
                    "area": {"cross": {"enable": False}, "embed": {"enable": False}, "softout": {"enable": False}},
                    "boundary": {"wall": {"enable": False}},
                    "focus": {"attract": {"enable": True}},
                    "anchor": {"spring": {"enable": True}},
                }
            }
        }
    }
    res4 = _solve_once(anchor_xy, wh, frame_size, args.config, overrides=o_af, return_viewpack=True)
    _print_run("anchor.spring + focus.attract", res4.viewpack, anchor_xy)

    if args.viz:
        from cartoweave.viz import show_vp
        print("\nOpening interactive viz for the last run (anchor + focus)...")
        show_vp(res4.viewpack, viz_cfg={"show": True, "panels": {"field": False}})


if __name__ == "__main__":
    main()
