import numpy as np
from cartoweave.engine.solvers.hybrid import solve_layout_hybrid


def _scene():
    return {
        "frame_size": (10.0, 10.0),
        "labels": [
            {"visible": True, "mode": "rect"},
            {"visible": True, "mode": "rect"},
        ],
        "labels_init": np.array([[0.0, 0.0], [0.1, 0.0]], float),
        "WH": np.ones((2, 2), float),
        "anchors": np.zeros((2, 2), float),
    }


def test_hybrid_solver_records_ginf():
    scene = _scene()
    cfg = {"ll.k.repulse": 10.0}
    _, info = solve_layout_hybrid(scene, cfg)
    recs = info.get("history", {}).get("records", [])
    assert recs, "no records recorded"
    si = recs[0]["meta"].get("solver_info")
    assert si is not None
    assert si["solver"] == "hybrid"
    assert si["gtol"] > 0
    assert si["g_inf"] > 0
