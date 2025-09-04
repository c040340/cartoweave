import numpy as np
import pytest
import numpy as np


def _tiny_scene():
    return dict(
        frame=0,
        frame_size=(640, 360),
        labels_init=np.array([[100.0, 120.0], [220.0, 180.0]], dtype=float),
        WH=np.array([[80.0, 28.0], [80.0, 28.0]], dtype=float),
        anchors=np.zeros((2, 2), dtype=float),
        labels=[{"anchor_kind": "none"}, {"anchor_kind": "none"}],
        points=[],
        lines=[],
        areas=[],
    )


def test_lbfgs_records_per_evaluation():
    from cartoweave.engine.solvers import lbfgs

    scene = _tiny_scene()
    P0 = np.asarray(scene["labels_init"], float)
    cfg = {
        "solver": {"lbfgs": {"maxiter": 20, "pgtol": 1e-1}},
        "anchor.k.spring": 10.0,
    }

    outer_calls = {"n": 0}

    def outer_rec(P, E, comps, meta):
        outer_calls["n"] += 1

    info = lbfgs.run(scene, P0, cfg, record=outer_rec)
    hist = info.get("history", {})
    recs = hist.get("records", [])
    assert len(recs) > 0, "No per-evaluation records captured"

    r0 = recs[0]
    assert "P" in r0 and "E" in r0 and "comps" in r0 and "meta" in r0
    assert r0["P"].shape == P0.shape == (2, 2)
    assert isinstance(r0["E"], float)
    assert isinstance(r0["comps"], dict) and len(r0["comps"]) >= 1
    for k, v in r0["comps"].items():
        v = np.asarray(v)
        assert v.shape == (2, 2), f"term '{k}' must be (N,2), got {v.shape}"

    assert outer_calls["n"] >= 1


@pytest.mark.optional
def test_records_include_stage_meta_if_plan_is_used():
    try:
        from cartoweave.orchestrators.solve_plan_runner import run_solve_plan
        from cartoweave.engine.solvers import lbfgs  # ensure it imports
    except Exception:
        pytest.skip("solve_plan orchestrator not available")

    scene = _tiny_scene()
    plan = [dict(name="warmup"), dict(name="main")]
    got_stage_ids = set()

    def rec(P, E, comps, meta):
        if meta and "stage_id" in meta:
            got_stage_ids.add(meta["stage_id"])

    run_solve_plan(scene, plan, cfg={}, record=rec)
    assert got_stage_ids == {0, 1}

