import numpy as np

def test_summary_has_schema_runtime_and_time():
    from cartoweave.contracts.solvepack import SolvePack
    from cartoweave.compute.run import solve
    L = 3
    scene = {
        "labels_init": np.zeros((L, 2)),
        "labels": [{"anchor_kind": "none", "anchor_index": -1}] * L,
        "frame_size": (1920, 1080),
        "WH": np.ones((L, 2)),
        "points": np.zeros((0, 2)),
        "lines": [],
        "areas": [],
    }
    sp = SolvePack(
        L=L,
        P0=np.zeros((L, 2)),
        active_mask0=np.ones(L, bool),
        scene=scene,
        cfg={"compute": {"weights": {"anchor.spring": 1.0}, "eps": {"numeric": 1e-12}}},
        stages=[{"iters": 1, "solver": "lbfgs"}],
        passes=["schedule", "capture"],
    )
    view = solve(sp)
    s = view.summary
    assert s.get("schema_version") == "v1"
    assert "runtime" in s and "cartoweave" in s["runtime"]
    assert isinstance(s.get("time_ms"), int)
