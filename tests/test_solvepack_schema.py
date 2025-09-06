import pytest
from cartoweave.contracts import SolvePack, Scene, Label, Anchor


def _valid_pack() -> SolvePack:
    scene = Scene(frame_size=(100.0, 50.0))
    labels = [
        Label(id=0, kind="point", anchor=Anchor(mode="xy", xy=(0.0, 0.0))),
        Label(id=1, kind="line", anchor=Anchor(mode="xy", xy=(1.0, 1.0)), polyline=[(0.0, 0.0), (1.0, 1.0)]),
    ]
    pack = SolvePack(
        L=2,
        P0=[(0.0, 0.0), (1.0, 1.0)],
        labels0=labels,
        active0=[True, False],
        scene0=scene,
        cfg={"compute": {"eps": 1e-9}},
    )
    return pack


def test_valid_pack_passes():
    _valid_pack()  # should not raise


def test_mismatched_lengths_raises():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [Label(id=0, kind="point", anchor=Anchor(mode="xy", xy=(0.0, 0.0)))]
    with pytest.raises(ValueError):
        SolvePack(
            L=2,
            P0=[(0.0, 0.0)],
            labels0=labels,
            active0=[True],
            scene0=scene,
            cfg={"compute": {}},
        )


def test_line_requires_polyline():
    scene = Scene(frame_size=(10.0, 10.0))
    # provide invalid label as raw dict to defer validation until SolvePack
    labels = [
        {
            "id": 0,
            "kind": "line",
            "anchor": {"mode": "xy", "xy": (0.0, 0.0)},
        }
    ]
    with pytest.raises(ValueError):
        SolvePack(
            L=1,
            P0=[(0.0, 0.0)],
            labels0=labels,  # type: ignore[arg-type]
            active0=[True],
            scene0=scene,
            cfg={"compute": {}},
        )


def test_cfg_extra_key_raises():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        Label(id=0, kind="point", anchor=Anchor(mode="xy", xy=(0.0, 0.0))),
        Label(id=1, kind="point", anchor=Anchor(mode="xy", xy=(1.0, 1.0))),
    ]
    with pytest.raises(ValueError):
        SolvePack(
            L=2,
            P0=[(0.0, 0.0), (1.0, 1.0)],
            labels0=labels,
            active0=[True, True],
            scene0=scene,
            cfg={"compute": {}, "extra": {}},
        )


def test_cfg_legacy_keys_raise():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        Label(id=0, kind="point", anchor=Anchor(mode="xy", xy=(0.0, 0.0))),
        Label(id=1, kind="point", anchor=Anchor(mode="xy", xy=(1.0, 1.0))),
    ]
    with pytest.raises(ValueError):
        SolvePack(
            L=2,
            P0=[(0.0, 0.0), (1.0, 1.0)],
            labels0=labels,
            active0=[True, True],
            scene0=scene,
            cfg={"compute": {}, "solver": {}},
        )


def test_label_id_mismatch_raises():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        Label(id=1, kind="point", anchor=Anchor(mode="xy", xy=(0.0, 0.0))),
        Label(id=0, kind="point", anchor=Anchor(mode="xy", xy=(1.0, 1.0))),
    ]
    with pytest.raises(ValueError):
        SolvePack(
            L=2,
            P0=[(0.0, 0.0), (1.0, 1.0)],
            labels0=labels,
            active0=[True, True],
            scene0=scene,
            cfg={"compute": {}},
        )


def test_area_requires_polygon():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        {
            "id": 0,
            "kind": "area",
            "anchor": {"mode": "xy", "xy": (0.0, 0.0)},
        }
    ]
    with pytest.raises(ValueError):
        SolvePack(
            L=1,
            P0=[(0.0, 0.0)],
            labels0=labels,  # type: ignore[arg-type]
            active0=[True],
            scene0=scene,
            cfg={"compute": {}},
        )


def test_active0_strict_bool():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        Label(id=0, kind="point", anchor=Anchor(mode="xy", xy=(0.0, 0.0))),
    ]
    with pytest.raises(ValueError):
        SolvePack(
            L=1,
            P0=[(0.0, 0.0)],
            labels0=labels,
            active0=[1],  # type: ignore[list-item]
            scene0=scene,
            cfg={"compute": {}},
        )


def test_point_rejects_polyline_polygon():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        {
            "id": 0,
            "kind": "point",
            "anchor": {"mode": "xy", "xy": (0.0, 0.0)},
            "polyline": [(0.0, 0.0), (1.0, 1.0)],
        }
    ]
    with pytest.raises(ValueError):
        SolvePack(
            L=1,
            P0=[(0.0, 0.0)],
            labels0=labels,  # type: ignore[arg-type]
            active0=[True],
            scene0=scene,
            cfg={"compute": {}},
        )
