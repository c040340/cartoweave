# ruff: noqa: S101
from cartoweave.compute.solve import solve
from cartoweave.data.api import make_solvepack_from_data_defaults


def test_pass_stats():
    compute_cfg = {
        "passes": {
            "nan_guard": {},
            "grad_clip": {"max_norm": 1.0},
            "step_limit": {"max_step_norm": 1.5},
            "capture": {"every": 1},
        }
    }
    sp = make_solvepack_from_data_defaults(compute_cfg=compute_cfg)
    vp = solve(sp)
    ps = vp.summary.get("pass_stats", {})
    assert ps.get("NaNGuardPass", {}).get("fixed_frames", 0) >= 0
    assert ps.get("GradClipPass", {}).get("clipped_frames", 0) >= 0
    assert ps.get("StepLimitPass", {}).get("clamped_steps", 0) >= 0
