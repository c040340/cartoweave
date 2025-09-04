# Phase A diagnostics (for context):
# - random_solve_plan example previously read ``info['scene_script']`` and exposed no
#   ``history['records']``; iteration slider could not show frames.
# - solver records lacked ``step_id`` tags, so action bar segmentation failed.
# - examples ignored solver histories entirely, showing only final positions.

import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure repository root (for examples package) is on the path when pytest
# tweaks ``sys.path`` via ``pythonpath`` in pytest.ini.
sys.path.append(str(Path(__file__).resolve().parents[1]))


def _tiny_scene():
    return dict(
        frame=0,
        frame_size=(640, 360),
        labels_init=np.array([[100.0, 120.0], [220.0, 180.0]], float),
        WH=np.array([[80.0, 28.0], [80.0, 28.0]], float),
        labels=[{"anchor_kind": "none", "id": "l0"}, {"anchor_kind": "none", "id": "l1"}],
        points=[],
        lines=[],
        areas=[],
    )


def test_minimal_fit_viz_bridge_smoke(monkeypatch):
    try:
        from cartoweave.engine.solvers import lbfgs
    except Exception:
        pytest.skip("solver not available")

    from examples.minimal_fit import build_viz_payload

    scene = _tiny_scene()
    P0 = scene["labels_init"].copy()
    cfg = {"solver": {"lbfgs": {"maxiter": 10, "pgtol": 5e-1}}, "anchor.k.spring": 10.0}

    info = lbfgs.run(scene, P0, cfg)
    payload = build_viz_payload(info)

    assert "frames" in payload and len(payload["frames"]) > 0, "No iteration frames for slider"
    f0 = payload["frames"][0]
    assert "P" in f0 and isinstance(f0["P"], np.ndarray) and f0["P"].shape == (2, 2)
    assert "comps" in f0 and isinstance(f0["comps"], dict) and len(f0["comps"]) >= 1
    any_term = next(iter(f0["comps"].values()))
    assert isinstance(any_term, np.ndarray) and any_term.shape == (2, 2)

    if "meta" in f0 and isinstance(f0["meta"], dict) and "stage_id" in f0["meta"]:
        assert isinstance(f0["meta"]["stage_id"], int)


def test_random_solve_plan_viz_bridge_smoke(monkeypatch):
    try:
        from cartoweave.api import solve_scene_script
        from examples.random_solve_plan import build_viz_payload, build_solve_plan
    except Exception:
        pytest.skip("solve_plan example not available")

    scene = _tiny_scene()
    script = {"steps": [{"name": "enter", "op": "enter", "id": "l0"}]}
    plan = {"stages": build_solve_plan(scene, cfg={})}
    info = solve_scene_script(scene, script, {"boundary.k.wall": 1.0}, solve_plan=plan)
    payload = build_viz_payload(info)

    assert "frames" in payload and len(payload["frames"]) > 0
    assert "steps" in payload and len(payload["steps"]) == 1
    first = payload["steps"][0]
    assert first.get("rec_start") == 0 and first.get("rec_end") == len(payload["frames"])
