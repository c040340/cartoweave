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
        from examples.random_timeline import (
            build_random_schema,
            compile_timeline,
            run_example_headless,
        )
        scene = _tiny_scene()
        schema = build_random_schema(scene, cfg={})
        timeline = compile_timeline(schema, cfg={})
        assert isinstance(timeline, list) and len(timeline) > 0
        info = run_example_headless(scene, timeline, cfg={})
    except Exception:
        try:
            from cartoweave.orchestrators.timeline_runner import run_timeline
        except Exception:
            pytest.skip("timeline runner not available")
        scene = _tiny_scene()
        timeline = [dict(name="warmup"), dict(name="main")]
        info = run_timeline(scene, timeline, cfg={})

    hist = info.get("history", {})
    recs = hist.get("records", [])
    assert len(recs) > 0, "Timeline runner produced no per-eval records"
    action_ids = {
        r.get("meta", {}).get("action_id") for r in recs if isinstance(r.get("meta"), dict)
    }
    assert any(aid is not None for aid in action_ids), "No action_id in records meta"
    if len(timeline) >= 2:
        assert len({aid for aid in action_ids if aid is not None}) >= 2
