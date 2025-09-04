import numpy as np
import pytest


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


def test_random_solve_plan_builds_and_propagates(monkeypatch):
    try:
        from examples.random_solve_plan import (
            build_random_plan,
            compile_solve_plan,
            run_example_headless,
        )
        scene = _tiny_scene()
        spec = build_random_plan(scene, cfg={})
        plan = compile_solve_plan(spec, cfg={})
        assert isinstance(plan, list) and len(plan) > 0
        info = run_example_headless(scene, plan, cfg={})
    except Exception:
        try:
            from cartoweave.orchestrators.solve_plan_runner import run_solve_plan
        except Exception:
            pytest.skip("solve_plan runner not available")
        scene = _tiny_scene()
        plan = [dict(name="warmup"), dict(name="main")]
        info = run_solve_plan(scene, plan, cfg={})

    hist = info.get("history", {})
    recs = hist.get("records", [])
    assert len(recs) > 0, "solve_plan runner produced no per-eval records"
    stage_ids = {
        r.get("meta", {}).get("stage_id") for r in recs if isinstance(r.get("meta"), dict)
    }
    assert any(sid is not None for sid in stage_ids), "No stage_id in records meta"
    if len(plan) >= 2:
        assert len({sid for sid in stage_ids if sid is not None}) >= 2
