from cartoweave.compute.run import solve
from cartoweave.contracts import SolvePack, Scene, Label, Anchor


def test_pipeline_smoke():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        Label(id=0, kind="point", anchor=Anchor(mode="xy", xy=(0.0, 0.0))),
        Label(id=1, kind="line", anchor=Anchor(mode="xy", xy=(0.0, 0.0)), polyline=[(0.0, 0.0), (1.0, 0.0)]),
        Label(id=2, kind="area", anchor=Anchor(mode="xy", xy=(0.0, 0.0)), polygon=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),
    ]
    pack = SolvePack(
        L=3,
        P0=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        labels0=labels,
        active0=[True, True, True],
        scene0=scene,
        cfg={"compute": {"weights": {"anchor.spring": 1.0}}},
    )
    pack.__dict__["stages"] = [{"iters": 1}]

    vp = solve(pack)
    assert len(vp.frames) >= 1
    assert len(vp.events) >= 1
    last = vp.last
    assert last.P.shape == (3, 2)
    for arr in last.comps.values():
        assert arr.shape == (3, 2)
