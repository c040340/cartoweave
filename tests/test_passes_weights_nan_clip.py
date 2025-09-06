from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def test_pass_stats():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 2, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {
            "passes": {
                "nan_guard": {},
                "grad_clip": {"max_norm": 1.0},
                "step_limit": {"max_step_norm": 1.5},
                "capture": {"every": 1},
            }
        },
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }
    sp = build_solvepack_from_config(cfg, seed=1)
    vp = solve(sp)
    ps = vp.summary.get("pass_stats", {})
    assert ps.get("NaNGuardPass", {}).get("fixed_frames", 0) >= 0
    assert ps.get("GradClipPass", {}).get("clipped_frames", 0) >= 0
    assert ps.get("StepLimitPass", {}).get("clamped_steps", 0) >= 0
