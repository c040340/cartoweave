import numpy as np

from cartoweave.contracts.viewpack_v1 import VPFrame


def test_conservation_active_only():
    N = 2
    frame = VPFrame(
        t=0,
        P=np.array([[0.0, 0.0], [np.nan, np.nan]]),
        comps={"f": np.array([[1.0, 2.0], [0.0, 0.0]])},
        E=0.0,
        active_mask=np.array([True, False]),
        meta={
            "global_iter": 0,
            "active_ids": [0],
            "active_count": 1,
            "events": [{"kind": "state_snapshot"}],
            "G_snapshot": np.array([[-1.0, -2.0], [np.nan, np.nan]]),
        },
        metrics={},
        field=None,
        anchors=np.zeros((N, 2)),
    )
    frame.validate(N)
