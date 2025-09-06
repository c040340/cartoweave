from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve_behaviors


def test_pipe_runs():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {
                "num_points": 2,
                "num_lines": 1,
                "num_areas": 1,
                "num_labels": 4,
                "num_steps": 6,
            },
        },
        "behavior": {
            "schedule_mode": "random_fair",
            "anchor_policy": "unique_first_then_rr",
        },
    }
    pack = build_solvepack_from_config(cfg, seed=7)
    state = solve_behaviors(pack)
    assert state.P.shape == (pack.N, 2)
    assert state.active.shape == (pack.N,)

