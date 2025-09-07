from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.compute.solve import solve


def test_recorder_produces_frames_and_events():
    pack = make_solvepack_from_data_defaults()
    vp = solve(pack)
    assert len(vp.frames) >= 1
    assert len(vp.events) >= 1
    summary = vp.summary
    assert summary["frames_captured"] == len(vp.frames)
    assert "terms_used" in summary
    assert "time_ms" in summary and summary["time_ms"] >= 0
    assert summary["global_iters"] >= len(vp.frames)

