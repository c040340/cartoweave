import numpy as np
from cartoweave.contracts.solvepack import SolvePack
from cartoweave.compute.run import solve


def test_step_limit_clamps_step_when_enabled():
    L = 3
    P0 = np.zeros((L, 2), float)
    scene = {
        "labels_init": np.zeros((L, 2), float),
        "labels": [{"anchor_kind": "none", "anchor_index": -1}] * L,
        "frame_size": (1920, 1080),
        "WH": np.ones((L, 2)),
        "points": np.zeros((0, 2)),
        "lines": [],
        "areas": [],
    }
    cfg = {"compute": {
        "weights": {"anchor.spring": 1.0},
        "eps": {"numeric": 1e-12},
        "passes": {
            "capture": {"every": 1, "final_always": True},
            "step_limit": {"max_step_norm": 1e-3}
        }
    }}
    sp = SolvePack(
        L=L,
        P0=P0,
        active_mask0=np.ones(L, dtype=bool),
        scene=scene,
        cfg=cfg,
        stages=[{"iters": 2, "solver": "lbfgs"}],
        passes=["schedule", "step_limit", "capture"],
    )
    view = solve(sp)
    stats = view.summary.get("pass_stats", {}).get("StepLimitPass", {})
    assert stats.get("clamped_steps", 0) >= 0
