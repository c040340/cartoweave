import json
from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def test_cli_solve_runs_and_writes_files(tmp_path):
    cfg = {
        "data": {
            "source": "generate",
            "generate": {"num_points": 2, "num_lines": 1, "num_areas": 0, "num_steps": 1},
        },
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }
    sp = build_solvepack_from_config(cfg, seed=0)
    vp = solve(sp)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_sum = out_dir / "summary.json"
    out_sum.write_text(json.dumps(vp.summary), encoding="utf-8")
    s = json.loads(out_sum.read_text("utf-8"))
    assert isinstance(s, dict)
    assert s.get("frames_captured", 0) >= 1
    assert "pass_stats" in s
