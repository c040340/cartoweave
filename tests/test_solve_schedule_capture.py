from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def test_schedule_and_capture():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 3, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {"passes": {"capture": {"every": 1}, "step_limit": {"max_step_norm": 1.5}}},
        "behaviors": [{"solver": "lbfgs", "iters": 2}],
    }
    sp = build_solvepack_from_config(cfg, seed=0)
    vp = solve(sp)
    assert len(vp.frames) >= 1
    assert "pass_stats" in vp.summary
    ps = vp.summary["pass_stats"].get("StepLimitPass", {})
    assert ps.get("clamped_steps", 0) >= 0

