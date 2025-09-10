# -*- coding: utf-8 -*-
"""force_grad_validate.py — compare analytic and numeric gradients for forces.

Run as a standalone diagnostic script:

    python scripts/force_grad_validate.py

It builds a small synthetic scene and checks each compute.force term against a
central-difference numerical gradient.  Terms whose maximum relative error
exceeds ``1e-6`` are reported as ``BAD``.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Make project modules importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from cartoweave.config.loader import load_compute_config
from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.contracts.solvepack import Scene, Label, Anchor

TERMS = [
    "pl.rect",
    "ln.rect",
    "ll.rect",
    "area.embed",
    "area.cross",
    "area.softout",
    "boundary.wall",
    "focus.attract",
    "anchor.spring",
]


def build_scene():
    """Assemble a small scene with points, lines and areas."""
    scene = Scene(
        frame_size=(200.0, 150.0),
        points=[(5.0, 5.0)],
        lines=[[(40.0, 80.0), (80.0, 80.0)]],
        areas=[
            [(80.0, 50.0), (100.0, 50.0), (100.0, 70.0), (80.0, 70.0)],
            [(110.0, 50.0), (130.0, 50.0), (130.0, 70.0), (110.0, 70.0)],
        ],
    )
    labels = [
        Label(id=0, kind="point", WH=(10.0, 10.0), anchor=Anchor(target="point", index=0, mode="exact")),
        Label(id=1, kind="line", WH=(10.0, 10.0), anchor=Anchor(target="line", index=0, mode="midpoint")),
        Label(id=2, kind="area", WH=(12.0, 12.0), anchor=Anchor(target="area", index=0, mode="centroid")),
        Label(id=3, kind="area", WH=(12.0, 12.0), anchor=Anchor(target="area", index=1, mode="centroid")),
    ]
    P = np.array(
        [
            [4.0, 8.0],
            [85.0, 55.0],
            [92.0, 62.0],
            [118.0, 62.0],
        ],
        dtype=float,
    )
    active = np.array([True, True, True, True])
    return scene, labels, P, active


def max_grad_error(term: str, scene, labels, P, active, h: float = 1e-4) -> float:
    """Return maximum relative error between analytic and numeric gradients."""
    cfg = load_compute_config(
        internals_path=str(ROOT / "configs/compute.internals.yaml"),
        tuning_path=str(ROOT / "configs/compute.tuning.yaml"),
        public_path=str(ROOT / "configs/compute.public.yaml"),
    )
    forces = cfg["compute"]["public"]["forces"]
    for group in forces.values():
        for params in group.values():
            if isinstance(params, dict):
                params["enable"] = False
    grp, name = term.split(".")
    forces[grp][name]["enable"] = True
    E0, g, _ = energy_and_grad_full(P, labels, scene, active, cfg["compute"])
    g_ana = g.reshape(-1)
    flatP = P.reshape(-1)
    max_err = 0.0
    for k in range(flatP.size):
        Pm = flatP.copy(); Pp = flatP.copy()
        Pm[k] -= h; Pp[k] += h
        Em, _g, _ = energy_and_grad_full(Pm.reshape(P.shape), labels, scene, active, cfg["compute"])
        Ep, _g, _ = energy_and_grad_full(Pp.reshape(P.shape), labels, scene, active, cfg["compute"])
        g_num = (Ep - Em) / (2 * h)
        denom = max(1.0, abs(g_num), abs(g_ana[k]))
        err = abs(g_ana[k] - g_num) / denom
        if err > max_err:
            max_err = err
    return max_err


def main():
    scene, labels, P, active = build_scene()
    tol = 1e-6
    for term in TERMS:
        err = max_grad_error(term, scene, labels, P, active)
        status = "OK" if err < tol else "BAD"
        print(f"{term:14s} err={err:.3e} -> {status}")


if __name__ == "__main__":
    main()
