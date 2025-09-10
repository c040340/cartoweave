import numpy as np

from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.contracts import Anchor, Label, Scene, SolvePack


def test_pipeline_smoke():
    scene = Scene(
        frame_size=(10.0, 10.0),
        points=[(0.0, 0.0)],
        lines=[[(0.0, 0.0), (1.0, 0.0)]],
        areas=[[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]],
    )
    labels = [
        Label(
            id=0,
            kind="point",
            anchor=Anchor(target="point", index=0, mode="exact"),
            meta={"anchor_xy": (0.0, 0.0)},
        ),
        Label(
            id=1,
            kind="line",
            anchor=Anchor(target="line", index=0, mode="midpoint"),
            meta={"anchor_xy": (0.5, 0.0)},
        ),
        Label(
            id=2,
            kind="area",
            anchor=Anchor(target="area", index=0, mode="centroid"),
            meta={"anchor_xy": (1.0 / 3.0, 1.0 / 3.0)},
        ),
    ]
    pack = SolvePack(
        L=3,
        P0=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        labels0=labels,
        active0=[True, True, True],
        scene0=scene,
        cfg={
            "compute": {
                "public": {
                    "forces": {
                        "anchor.spring": {
                            "enable": True,
                            "k_anchor_spring": 1.0,
                        }
                    }
                }
            }
        },
    )
    p = np.asarray(pack.P0, float)
    active = np.asarray(pack.active0, bool)
    scene_dict = pack.scene0.model_dump()
    _, g, comps = energy_and_grad_full(p, pack.labels0, scene_dict, active, pack.cfg["compute"])
    assert g.shape == (3, 2)  # noqa: S101
    for arr in comps.values():
        assert arr.shape == (3, 2)  # noqa: S101
