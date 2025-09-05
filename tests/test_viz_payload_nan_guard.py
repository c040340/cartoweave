import numpy as np

from cartoweave.viz.build_viz_payload import build_viz_payload


def test_build_viz_payload_sanitizes_nan() -> None:
    info = {
        "scene": {"labels": [{}], "WH": np.ones((1, 2))},
        "history": {
            "records": [
                {
                    "P": np.array([[np.nan, 0.0]]),
                    "comps": {"foo": np.array([[np.inf, -np.inf]])},
                    "meta": {},
                }
            ],
            "positions": [np.array([[np.nan, 0.0]])],
            "energies": [0.0],
        },
    }
    payload = build_viz_payload(info)
    frm = payload["frames"][0]
    assert np.isfinite(frm["P"]).all()
    for arr in frm["comps"].values():
        assert np.isfinite(arr).all()
