import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure repository root on sys.path for "examples" package.
sys.path.append(str(Path(__file__).resolve().parents[1]))


def _fake_history_with_steps():
    recs = []
    steps = []

    def mk_frame(idx, sid):
        P = np.array([[0.0, 0.0], [1.0, 1.0]], float)
        comps = {"focus": np.zeros_like(P), "ll": np.ones_like(P)}
        meta = {"step_id": sid, "step_name": f"a{sid}", "eval_index": idx}
        return {"P": P, "E": 0.0, "comps": comps, "meta": meta}

    for i in range(3):
        recs.append(mk_frame(i, 0))
    steps.append({"name": "a0", "rec_start": 0, "rec_end": 3})
    for i in range(3, 6):
        recs.append(mk_frame(i, 1))
    steps.append({"name": "a1", "rec_start": 3, "rec_end": 6})
    return {"history": {"records": recs, "scene_steps": steps}}


def test_step_segments_and_drag_mapping():
    try:
        from examples.minimal_fit import build_viz_payload
    except Exception:
        pytest.skip("payload builder not available in examples")
    info = _fake_history_with_steps()
    payload = build_viz_payload(info)
    segs = payload.get("steps") or []
    assert isinstance(segs, list) and len(segs) >= 2
    ids = {s.get("id") for s in segs}
    assert ids == {0, 1}
    get_idx = payload.get("get_frame_index_for_step")
    assert callable(get_idx)
    assert get_idx(0) == 0
    assert get_idx(1) == 3


def test_right_panel_payload_has_terms_and_label_selection():
    try:
        from examples.minimal_fit import build_viz_payload
    except Exception:
        pytest.skip("payload builder not available")
    info = _fake_history_with_steps()
    payload = build_viz_payload(info)
    frames = payload.get("frames") or payload.get("evals") or []
    assert len(frames) > 0
    f0 = frames[0]
    assert "P" in f0 and getattr(f0["P"], "shape", None) == (2, 2)
    comps = f0.get("comps", {})
    assert isinstance(comps, dict) and len(comps) >= 1
    known_or_aliased = any(k in comps for k in ("focus", "ll", "boundary", "anchor", "total"))
    assert known_or_aliased
    sl = payload.get("selected_label", 0)
    assert isinstance(sl, int) and sl >= 0
