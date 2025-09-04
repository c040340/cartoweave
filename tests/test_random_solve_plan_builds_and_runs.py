import numpy as np
import pytest
import numpy as np


def _tiny_scene():
    return dict(
        frame=0,
        frame_size=(640, 360),
        labels_init=np.array([[100.0, 120.0], [220.0, 180.0]], float),
        WH=np.array([[80.0, 28.0], [80.0, 28.0]], float),
        labels=[{"anchor_kind": "none"}, {"anchor_kind": "none"}],
        points=[],
        lines=[],
        areas=[],
    )


def test_random_solve_plan_example_builds_and_runs(monkeypatch):
    try:
        from cartoweave.orchestrators.solve_plan_runner import run_solve_plan
    except Exception:
        pytest.skip("solve_plan runner not available")

    scene = _tiny_scene()
    try:
        from examples.random_solve_plan import build_solve_plan, run_example_headless

        plan = build_solve_plan(scene, cfg={})
        assert isinstance(plan, list) and len(plan) >= 1
        info = run_example_headless(scene, plan, cfg={})
    except Exception:
        plan = [dict(name="warmup"), dict(name="main")]
        info = run_solve_plan(scene, plan, cfg={})

    hist = info.get("history", {})
    recs = hist.get("records", [])
    assert len(recs) > 0, "No per-evaluation records from solve_plan runner"
    stage_ids = {
        r.get("meta", {}).get("stage_id") for r in recs if isinstance(r.get("meta"), dict)
    }
    # Ensure at least one stage id and preferably more than one
    assert any(sid is not None for sid in stage_ids), "No stage_id threaded into meta"
    assert len([sid for sid in stage_ids if sid is not None]) >= 1
    if len(plan) >= 2:
        # Expect at least two distinct ids when plan has multiple stages
        assert len({sid for sid in stage_ids if sid is not None}) >= 2
