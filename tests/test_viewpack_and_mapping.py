from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def test_viewpack_basic():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {
                "num_points": 2,
                "num_lines": 0,
                "num_areas": 0,
                "num_steps": 1,
            },
        },
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }
    sp = build_solvepack_from_config(cfg, seed=0)
    vp = solve(sp)
    assert isinstance(vp.frames, list) and len(vp.frames) >= 1
    assert vp.last is vp.frames[-1]
    assert "pass_stats" in vp.summary
    assert isinstance(vp.terms_used, list)
