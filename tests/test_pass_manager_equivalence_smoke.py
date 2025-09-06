import numpy as np
from cartoweave.contracts.solvepack import SolvePack
from cartoweave.compute.run import solve


def test_pass_manager_equivalence_smoke(scene):
    L = len(scene["labels"])
    P0 = np.zeros((L, 2), float)
    scene = {**scene, "labels_init": np.zeros((L, 2), float)}
    cfg = {"compute": {"weights": {"anchor.spring": 1.0}, "eps": {"numeric": 1e-12},
                        "passes": {"capture": {"every": 1, "final_always": True}}}}
    sp = SolvePack(
        L=L,
        P0=P0,
        active_mask0=np.ones(L, dtype=bool),
        scene=scene,
        cfg=cfg,
        stages=[{"iters": 4, "solver": "lbfgs"}],
        passes=["schedule", "capture"],
    )
    view = solve(sp)
    assert view.summary["frames_captured"] >= 1
    assert "pass_stats" in view.summary
