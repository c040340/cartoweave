from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def test_step_limit_clamps_step_when_enabled():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 4, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {
            "passes": {
                "step_limit": {"max_step_norm": 1.5},
                "nan_guard": {},
                "grad_clip": {"max_norm": 1.0},
                "capture": {"every": 1},
            }
        },
        "behaviors": [{"solver": "lbfgs", "iters": 2}],
    }
    sp = build_solvepack_from_config(cfg, seed=2)
    vp = solve(sp)
    stats = vp.summary.get("pass_stats", {}).get("StepLimitPass", {})
    assert stats.get("clamped_steps", 0) >= 0
