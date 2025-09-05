import os
import sys
import numpy as np
import importlib
import cartoweave.compute.forces as new_forces
from cartoweave.data.random import generate_scene
from cartoweave.utils.geometry import polylines_to_segments

# Ensure legacy forces package importable
LEGACY_DIR = os.environ.get("LEGACY_FORCES_DIR")
if LEGACY_DIR is None:
    raise RuntimeError("LEGACY_FORCES_DIR not set")
PARENT = os.path.dirname(os.path.abspath(LEGACY_DIR))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import legacy_forces as legacy_forces

TERMS = [
    ("pl.rect", "pre_anchor"),
    ("ln.rect", "pre_anchor"),
    ("ll.rect", "pre_anchor"),
    ("ll.disk", "pre_anchor"),
    ("area.embed", "pre_anchor"),
    ("boundary.wall", "pre_anchor"),
    ("focus.attract", "pre_anchor"),
    ("anchor.spring", "anchor"),
    ("area.softout", "pre_anchor"),
    ("area.cross", "pre_anchor"),
]


def _build_cfg():
    w = {name: 1.0 for name, _ in TERMS}
    cfg = {
        "compute": {"weights": w},
        "pl.k.repulse": 1.0,
        "pl.k.inside": 0.5,
        "ln.k.repulse": 1.0,
        "ln.k.inside": 0.5,
        "ll.k.repulse": 1.0,
        "ll.k.inside": 0.5,
        "area.k.embed": 1.0,
        "area.k.cross": 1.0,
        "area.k.softout": 1.0,
        "boundary.k.wall": 1.0,
        "focus.k.attract": 1.0,
        "anchor.k.spring": 1.0,
    }
    return cfg


def _fd_check(fn, scene, P, cfg, phase, F, n=5, h=1e-6):
    L = P.shape[0]
    idxs = np.random.default_rng(0).choice(L, size=min(n, L), replace=False)
    for i in idxs:
        for d in range(2):
            dP = np.zeros_like(P)
            dP[i, d] = h
            E_plus, _, _ = fn(scene, P + dP, cfg, phase)
            E_minus, _, _ = fn(scene, P - dP, cfg, phase)
            fd = (E_plus - E_minus) / (2 * h)
            assert abs(fd + F[i, d]) <= 2e-4


def test_force_parity():
    scene = generate_scene(n_points=3, n_lines=2, n_areas=2, seed=0)
    scene["lines"] = polylines_to_segments(scene["lines"]).reshape(-1, 4)
    P = np.asarray(scene["labels_init"], float)
    cfg = _build_cfg()
    for name, phase in TERMS:
        fn_new = new_forces.REGISTRY[name]
        fn_old = legacy_forces.REGISTRY[name]
        E_new, F_new, _ = fn_new(scene, P, cfg, phase)
        E_old, F_old, _ = fn_old(scene, P, cfg, phase)
        assert abs(E_new - E_old) <= 1e-8 * max(1.0, abs(E_old))
        assert np.max(np.abs(F_new - F_old)) <= 1e-6
        _fd_check(fn_new, scene, P, cfg, phase, F_new)
