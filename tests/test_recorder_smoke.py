from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.compute.solve import solve


def test_recorder_produces_frames_and_events():
    pack = make_solvepack_from_data_defaults()
    vp = solve(pack)
    assert vp.num_frames() >= 1
    events = [ev for fr in vp.frames for ev in fr.meta["events"]]
    assert events
    assert vp.passes and vp.passes[-1].t_end == vp.num_frames()

