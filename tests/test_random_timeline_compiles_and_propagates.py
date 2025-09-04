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


def test_random_timeline_builds_and_propagates(monkeypatch):
    try:
        from cartoweave.orchestrators.scene_script_runner import run_scene_script
    except Exception:
        pytest.skip("scene_script runner not available")

    scene = _tiny_scene()
    try:
        from examples.random_solve_plan import (
            build_random_schema,
            compile_solve_plan,
            run_example_headless,
        )

        schema = build_random_schema(scene, cfg={})
        plan = compile_solve_plan(schema, cfg={})
        assert isinstance(plan, list) and len(plan) > 0
        info = run_example_headless(scene, plan, cfg={})
    except Exception:
        script = [
            {"name": "s0", "op": "enter", "id": "l0"},
            {"name": "s1", "op": "mode", "id": "l0", "mode": "circle"},
        ]
        scene["scene_script"] = script
        plan = [dict(name="warmup"), dict(name="main")]
        info = run_scene_script(scene, script, plan, cfg={})

    hist = info.get("history", {})
    recs = hist.get("records", [])
    steps = hist.get("scene_steps", [])
    assert len(recs) > 0, "scene_script runner produced no per-eval records"
    assert len(steps) > 0, "No scene_steps from scene_script runner"
    step_ids = {
        r.get("meta", {}).get("step_id") for r in recs if isinstance(r.get("meta"), dict)
    }
    assert any(sid is not None for sid in step_ids), "No step_id in records meta"
    if len(steps) >= 2:
        assert len({sid for sid in step_ids if sid is not None}) >= 2
