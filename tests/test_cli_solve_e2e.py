import json, subprocess, sys, pathlib, os


def test_cli_solve_runs_and_writes_files(tmp_path):
    root = pathlib.Path(".")
    cfg = root / "examples/configs/compute_min.json"
    scn = root / "examples/scenes/scene_min.json"
    out_sum = tmp_path / "summary.json"
    out_P = tmp_path / "P.json"

    cmd = [sys.executable, "-m", "cartoweave", "solve",
           "--config", str(cfg),
           "--scene", str(scn),
           "--out-summary", str(out_sum),
           "--out-P", str(out_P)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=root, env=env)
    assert cp.returncode == 0, cp.stderr

    assert out_sum.exists() and out_P.exists()
    s = json.loads(out_sum.read_text(encoding="utf-8"))
    assert s.get("frames_captured", 0) >= 1
