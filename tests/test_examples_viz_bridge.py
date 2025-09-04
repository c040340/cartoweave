# Phase A diagnostics (for context):
# - random_timeline example previously read ``info['timeline']`` and exposed no
#   ``history['records']``; iteration slider could not show frames.
# - solver records lacked ``action_id`` tags, so action bar segmentation failed.
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
        labels=[{"anchor_kind": "none"}, {"anchor_kind": "none"}],
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

    if "meta" in f0 and isinstance(f0["meta"], dict) and "action_id" in f0["meta"]:
        assert isinstance(f0["meta"]["action_id"], int)


def test_random_timeline_viz_bridge_smoke(monkeypatch):
    try:
        from cartoweave.orchestrators.timeline_runner import run_timeline
        from examples.random_timeline import build_viz_payload
    except Exception:
        pytest.skip("timeline example not available")

    tl = [dict(name="warmup"), dict(name="main")]
    info = run_timeline(_tiny_scene(), tl, cfg={})
    payload = build_viz_payload(info)

    assert "frames" in payload and len(payload["frames"]) > 0
    assert any(
        isinstance(f.get("meta"), dict) and "action_id" in f["meta"]
        for f in payload["frames"]
    ), "No action_id found in frames meta"
