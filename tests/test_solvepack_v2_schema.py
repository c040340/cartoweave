import pytest

from cartoweave.contracts import Anchor, Label, Scene, SolvePack


def _valid_pack() -> SolvePack:
    scene = Scene(
        frame_size=(100.0, 50.0),
        points=[(0.0, 0.0)],
        lines=[[(0.0, 0.0), (1.0, 0.0)]],
        areas=[[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]],
    )
    labels = [
        Label(id=0, kind="point", anchor=Anchor(target="point", index=0, mode="exact")),
        Label(id=1, kind="line", anchor=Anchor(target="line", index=0, mode="midpoint")),
        Label(id=2, kind="area", anchor=Anchor(target="area", index=0, mode="centroid")),
    ]
    pack = SolvePack(
        L=3,
        P0=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        labels0=labels,
        active0=[True, True, True],
        scene0=scene,
        cfg={"compute": {}},
    )
    return pack


def test_valid_pack_passes():
    _valid_pack()  # should not raise


def test_label_geometry_rejected():
    scene = Scene(frame_size=(10.0, 10.0))
    bad_label_polyline = {
        "id": 0,
        "kind": "point",
        "anchor": {"target": "free", "mode": "xy", "xy": (0.0, 0.0)},
        "polyline": [(0.0, 0.0), (1.0, 1.0)],
    }
    with pytest.raises(ValueError, match="Label must not carry geometry"):
        SolvePack(
            L=1,
            P0=[(0.0, 0.0)],
            labels0=[bad_label_polyline],
            active0=[True],
            scene0=scene,
            cfg={"compute": {}},
        )

    bad_label_polygon = {
        "id": 0,
        "kind": "point",
        "anchor": {"target": "free", "mode": "xy", "xy": (0.0, 0.0)},
        "polygon": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    }
    with pytest.raises(ValueError, match="Label must not carry geometry"):
        SolvePack(
            L=1,
            P0=[(0.0, 0.0)],
            labels0=[bad_label_polygon],
            active0=[True],
            scene0=scene,
            cfg={"compute": {}},
        )


def test_cfg_extra_key_raises():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        Label(id=0, kind="point", anchor=Anchor(target="free", mode="xy", xy=(0.0, 0.0)))
    ]
    with pytest.raises(ValueError):
        SolvePack(
            L=1,
            P0=[(0.0, 0.0)],
            labels0=labels,
            active0=[True],
            scene0=scene,
            cfg={"compute": {}, "extra": {}},
        )


def test_l_independent_from_geometry_counts():
    scene = Scene(
        frame_size=(10.0, 10.0),
        points=[(0.0, 0.0), (1.0, 1.0)],
        lines=[[(0.0, 0.0), (1.0, 0.0)]],
        areas=[],
    )
    labels = [
        Label(id=0, kind="point", anchor=Anchor(target="point", index=0, mode="exact"))
    ]
    SolvePack(
        L=1,
        P0=[(0.0, 0.0)],
        labels0=labels,
        active0=[True],
        scene0=scene,
        cfg={"compute": {}},
    )  # should not raise

