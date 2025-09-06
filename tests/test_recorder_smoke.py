from cartoweave.compute.run import solve
from cartoweave.contracts.solvepack import Scene, Label, SolvePack


def _pack_with_two_stages():
    lbl = Label(id=0, kind="point", anchor={"mode": "xy", "xy": (0.0, 0.0)}, meta={})
    pack = SolvePack(
        L=1,
        P0=[(0.0, 0.0)],
        labels0=[lbl],
        active0=[True],
        scene0=Scene(frame_size=(10.0, 10.0)),
        cfg={"compute": {}},
    )
    pack.__dict__["stages"] = [{}, {}]
    return pack


def test_recorder_produces_frames_and_events():
    pack = _pack_with_two_stages()
    vp = solve(pack)
    assert len(vp.frames) >= 2
    assert len(vp.events) >= 2
    summary = vp.summary
    assert summary["frames_captured"] == len(vp.frames)
    assert "terms_used" in summary
    assert "time_ms" in summary and summary["time_ms"] >= 0
    assert summary["global_iters"] >= len(vp.frames)

