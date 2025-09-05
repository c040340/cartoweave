import numpy as np
import logging
from cartoweave.orchestrators.solve_plan import run_solve_plan

def make_scene():
    return dict(
        frame=0,
        frame_size=(800, 600),
        labels_init=np.array([[200.0, 150.0], [260.0, 150.0]], float),
        WH=np.array([[60.0, 24.0], [60.0, 24.0]], float),
        labels=[{"anchor_kind": "none"}, {"anchor_kind": "none"}],
        points=np.zeros((0, 2), float),
        lines=np.zeros((0, 2, 2), float),
        areas=[],
        anchors=np.array([[180.0, 150.0], [220.0, 150.0]], float),
    )

def _base_cfg():
    return {
        "ll.geom": "rect",
        "ll.k.repulse": 150.0,
        "ll.k.inside": 50.0,
        "boundary.k.wall": 80.0,
        "boundary.wall_eps": 0.3,
        "anchor.k.spring": 10.0,
        "focus.k.attract": 30.0,
        "focus.center": np.array([400.0, 300.0]),
        "focus.sigma.x": 80.0,
        "focus.sigma.y": 120.0,
        "focus.delta": 10.0,
        "pl.k.repulse": 0.0,
        "pl.k.inside": 0.0,
    }

def test_injects_warmup_when_flag_true(caplog):
    scene = make_scene()
    cfg = _base_cfg()
    cfg["solver"] = {"public": {"use_warmup": True}, "tuning": {"warmup": {"step_cap_px": 5.0}}}
    plan = [{"name": "main"}]
    with caplog.at_level(logging.INFO):
        _, info = run_solve_plan(scene, cfg, plan, mode="hybrid", carry_P=True)
    assert info["solve_plan"][0]["name"] == "warmup_no_anchor"
    assert any("warmup_no_anchor" in r.message for r in caplog.records)

def test_no_injection_with_custom_stages(caplog):
    scene = make_scene()
    cfg = _base_cfg()
    cfg["solver"] = {"public": {"use_warmup": True}}
    plan = [{"name": "stage0"}, {"name": "stage1"}]
    with caplog.at_level(logging.INFO):
        _, info = run_solve_plan(scene, cfg, plan, mode="hybrid", carry_P=True)
    assert info["solve_plan"][0]["name"] == "stage0"
    assert not any("warmup_no_anchor" in r.message for r in caplog.records)
