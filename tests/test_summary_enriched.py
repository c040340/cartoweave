import numpy as np
from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def test_summary_has_schema_runtime_and_time():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 3, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }
    sp = build_solvepack_from_config(cfg, seed=0)
    view = solve(sp)
    s = view.summary
    assert s.get("frames_captured", 0) >= 1
    assert "pass_stats" in s
