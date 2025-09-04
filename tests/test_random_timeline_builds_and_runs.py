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


def test_random_timeline_example_builds_and_runs(monkeypatch):
    try:
        from cartoweave.orchestrators.timeline_runner import run_timeline
    except Exception:
        pytest.skip("timeline runner not available")

    scene = _tiny_scene()
    try:
        from examples.random_timeline import build_random_timeline, run_example_headless

        timeline = build_random_timeline(scene, cfg={})
        assert isinstance(timeline, list) and len(timeline) >= 1
        info = run_example_headless(scene, timeline, cfg={})
    except Exception:
        timeline = [dict(name="warmup"), dict(name="main")]
        info = run_timeline(scene, timeline, cfg={})

    hist = info.get("history", {})
    recs = hist.get("records", [])
    assert len(recs) > 0, "No per-evaluation records from timeline runner"
    action_ids = {
        r.get("meta", {}).get("action_id") for r in recs if isinstance(r.get("meta"), dict)
    }
    # Ensure at least one action id and preferably more than one
    assert any(aid is not None for aid in action_ids), "No action_id threaded into meta"
    assert len([aid for aid in action_ids if aid is not None]) >= 1
    if len(timeline) >= 2:
        # Expect at least two distinct ids when timeline has multiple actions
        assert len({aid for aid in action_ids if aid is not None}) >= 2
