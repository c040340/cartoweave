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
    events = [
        e for fr in vp.frames for e in fr.meta["events"] if e.get("pass")
    ]
    assert events  # passes emitted events
