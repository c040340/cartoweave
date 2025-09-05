import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from cartoweave.viz.build_viz_payload import build_viz_payload
from cartoweave.viz.panels import draw_layout
from cartoweave.viz.defaults import merge_defaults


def test_circle_anchor_and_nan_repair() -> None:
    scene = {
        "labels": [
            {"mode": "circle", "anchor_kind": "point", "anchor_index": 0},
            {},
        ],
        "labels_init": np.array([[5.0, 5.0], [1.0, 1.0]], float),
        "points": np.array([[2.0, 2.0]], float),
        "WH": np.ones((2, 2), float) * 2.0,
    }
    history = {
        "scene_steps": [
            {
                "active_ids_viz": [0, 1],
                "active_ids_solver": [1],
                "rec_start": 0,
                "rec_end": 2,
            }
        ],
        "records": [
            {"P": np.array([[1.0, 1.0]]), "comps": {}, "meta": {}},
            {"P": np.array([[np.nan, np.nan]]), "comps": {}, "meta": {}},
        ],
    }
    payload = build_viz_payload({"scene": scene, "history": history})
    frames = payload["frames"]
    assert frames[0]["P"].shape == (2, 2)
    assert np.allclose(frames[1]["P"][1], [1.0, 1.0])
    assert frames[0]["active_ids_solver"] == [1]
    assert frames[0]["active_ids_viz"] == [0, 1]

    anchors = np.array([[2.0, 2.0], [np.nan, np.nan]], float)
    fig, ax = plt.subplots()
    patches = draw_layout(
        ax,
        frames[0]["P"],
        scene["labels"],
        scene["WH"],
        frame_w=10,
        frame_h=10,
        points=None,
        lines=None,
        areas=None,
        anchors=anchors,
        viz_layout=merge_defaults(None)["layout"],
    )
    circ = None
    for idx, patch in patches:
        if idx == 0:
            circ = patch
    assert circ is not None and isinstance(circ, Circle)
    assert np.allclose(circ.center, anchors[0])
    plt.close(fig)
