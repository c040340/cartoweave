import numpy as np

from cartoweave.compute.recorder import ViewRecorder
from cartoweave.compute.types import VPSources


def test_view_recorder_builds_valid_viewpack():
    rec = ViewRecorder()
    N = 1
    labels = [{}]
    sources = VPSources(points=np.zeros((0, 2)), lines=[], areas=[], frame_size=(10, 10))
    rec.start_run(N, labels, None, sources, defaults={}, aux={})
    rec.start_pass(0, "main")
    rec.record_frame(
        t=0,
        P_full=np.zeros((N, 2)),
        comps_full={},
        E=0.0,
        active_mask=np.array([True]),
        anchors=np.zeros((N, 2)),
        meta_base={},
        metrics={},
        field=None,
        G_snapshot=None,
    )
    rec.end_pass()
    vp = rec.finish()
    assert vp.schema_version == "viewpack-v1"
    assert vp.num_frames() == 1
    assert vp.frames[0].meta["events"][0]["kind"] == "state_snapshot"
