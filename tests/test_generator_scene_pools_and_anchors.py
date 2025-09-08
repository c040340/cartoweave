from typing import Optional

import numpy as np

# ruff: noqa: S101, N803, N806
from cartoweave.config.loader import load_data_defaults
from cartoweave.contracts.solvepack import SolvePack
from cartoweave.data.generate import generate_scene


def _make_gen(points: int, lines: int, areas: int, labels: Optional[int]):
    cfg = load_data_defaults("configs/data.yaml")
    gen = cfg.generate
    assert gen is not None
    gen = gen.model_copy()
    gen.counts = gen.counts.model_copy(
        update={"points": points, "lines": lines, "areas": areas}
    )
    gen.labels = labels
    return gen


def _build_pack(P0, labels0, active0, scene0):
    return SolvePack(
        L=len(labels0),
        P0=[(float(x), float(y)) for x, y in P0],
        labels0=labels0,
        active0=active0.tolist(),
        scene0=scene0,
        cfg={"compute": {}},
    )


def test_default_label_count_and_anchor_refs():
    gen = _make_gen(5, 1, 1, labels=None)
    rng = np.random.default_rng(0)
    P0, labels0, scene0 = generate_scene(gen, rng)
    active0 = np.ones(len(labels0), dtype=bool)
    assert len(scene0.points) == 5
    assert len(scene0.lines) == 1
    assert len(scene0.areas) == 1
    assert len(labels0) == 7
    for lbl in labels0:
        t = lbl.anchor.target
        if t == "free":
            assert lbl.anchor.xy is not None
        elif t == "point":
            assert 0 <= lbl.anchor.index < 5
        elif t == "line":
            assert 0 <= lbl.anchor.index < 1
        elif t == "area":
            assert 0 <= lbl.anchor.index < 1
        assert lbl.ll_kernel == "rect"
    _build_pack(P0, labels0, active0, scene0)


def test_labels_exceed_geometry():
    gen = _make_gen(5, 1, 1, labels=10)
    rng = np.random.default_rng(0)
    P0, labels0, scene0 = generate_scene(gen, rng)
    active0 = np.ones(len(labels0), dtype=bool)
    assert len(labels0) == 10
    _build_pack(P0, labels0, active0, scene0)


def test_labels_fewer_than_geometry():
    gen = _make_gen(5, 1, 1, labels=3)
    rng = np.random.default_rng(0)
    P0, labels0, scene0 = generate_scene(gen, rng)
    active0 = np.ones(len(labels0), dtype=bool)
    assert len(labels0) == 3
    _build_pack(P0, labels0, active0, scene0)


def test_labels_do_not_carry_geometry():
    gen = _make_gen(2, 1, 1, labels=None)
    rng = np.random.default_rng(0)
    _, labels0, _ = generate_scene(gen, rng)
    for lbl in labels0:
        assert not hasattr(lbl, "polyline")
        assert not hasattr(lbl, "polygon")

