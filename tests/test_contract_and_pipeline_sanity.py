# ruff: noqa: S101, N803, N806
from pathlib import Path

import numpy as np
import pytest

from cartoweave.compute import solve
from cartoweave.config.loader import load_data_defaults
from cartoweave.contracts import Scene, SolvePack
from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.data.generate import generate_scene


def _make_gen(points: int, lines: int, areas: int, labels: int | None):
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


def test_schema_guards():
    scene = Scene(frame_size=(1.0, 1.0), points=[(0.0, 0.0)])
    lbl_bad = {
        "id": 0,
        "kind": "area",
        "anchor": {"target": "free", "mode": "xy", "xy": (0.0, 0.0)},
        "polygon": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    }
    with pytest.raises(ValueError, match="Label must not carry geometry"):
        SolvePack(
            L=1,
            P0=[(0.0, 0.0)],
            labels0=[lbl_bad],
            active0=[True],
            scene0=scene,
            cfg={"compute": {}},
        )

    lbl_ok = {
        "id": 0,
        "kind": "point",
        "anchor": {"target": "point", "index": 0, "mode": "exact"},
    }
    with pytest.raises(ValueError, match="cfg top-level keys must be"):
        SolvePack(
            L=1,
            P0=[(0.0, 0.0)],
            labels0=[lbl_ok],
            active0=[True],
            scene0=scene,
            cfg={"compute": {}, "data": {}},
        )


def test_generator_behavior_and_anchor_reuse():
    rng = np.random.default_rng(0)
    gen = _make_gen(5, 1, 1, labels=None)
    P0, labels0, scene0 = generate_scene(gen, rng)
    active0 = np.ones(len(labels0), dtype=bool)
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
    _build_pack(P0, labels0, active0, scene0)

    gen.labels = 10
    P0, labels0, scene0 = generate_scene(gen, rng)
    active0 = np.ones(len(labels0), dtype=bool)
    assert len(labels0) == 10
    anchors = [
        (lbl.anchor.target, lbl.anchor.index)
        for lbl in labels0
        if lbl.anchor.target != "free"
    ]
    assert len(anchors) > len(set(anchors))
    _build_pack(P0, labels0, active0, scene0)

    gen.labels = 3
    P0, labels0, scene0 = generate_scene(gen, rng)
    active0 = np.ones(len(labels0), dtype=bool)
    assert len(labels0) == 3
    _build_pack(P0, labels0, active0, scene0)


def test_end_to_end_solve_and_no_label_geometry_refs():
    pack = make_solvepack_from_data_defaults("configs/data.yaml")
    view = solve(pack)
    assert view.num_frames() >= 1
    for arr in view.frames[-1].comps.values():
        assert arr.shape == (pack.L, 2)
        assert np.isfinite(arr).all()

    src_dir = Path("src/cartoweave")
    offenders = []
    for path in src_dir.rglob("*.py"):
        text = path.read_text()
        if "label.polyline" in text or "label.polygon" in text:
            offenders.append(str(path))
    assert not offenders, f"found legacy geometry access in: {offenders}"

