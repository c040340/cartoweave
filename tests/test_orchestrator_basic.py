import numpy as np
from cartoweave.orchestrators.timeline import run_timeline

def make_scene():
    return dict(
        frame=0, frame_size=(800, 600),
        labels_init=np.array([[200.,150.],[260.,150.]], float),
        WH=np.array([[60.,24.],[60.,24.]], float),
        labels=[{"anchor_kind":"none"},{"anchor_kind":"none"}],
        points=np.zeros((0,2), float),
        lines=np.zeros((0,4), float),
        areas=[],
        anchors=np.array([[180.,150.],[220.,150.]], float),
    )

def test_timeline_runs_and_records():
    scene = make_scene()
    cfg = {
        "ll.geom": "rect", "ll.k.repulse": 150.0, "ll.k.inside": 50.0,
        "boundary.k.wall": 80.0, "boundary.wall_eps": 0.3,
        "anchor.k.spring": 10.0,
        "focus.k.attract": 30.0,
        "focus.center": np.array([400.,300.]),
        "focus.sigma.x": 80.0, "focus.sigma.y": 120.0,
        "focus.delta": 10.0,
        "pl.k.repulse": 0.0, "pl.k.inside": 0.0,  # 关闭点↔label 力
    }
    schedule = [
        {"name": "warmup_no_anchor", "scale": {"anchor.k.spring": 0.0}},
        {"name": "main_solve"},
    ]
    P_final, info = run_timeline(scene, cfg, schedule, mode="hybrid", carry_P=True)
    assert P_final.shape == scene["labels_init"].shape
    assert "timeline" in info and len(info["timeline"]) == 2
    # 最终解一般会有变化
    assert float(np.max(np.abs(P_final - scene["labels_init"]))) > 0.0
