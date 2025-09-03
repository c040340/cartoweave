import numpy as np
from cartoweave.orchestrators.timeline import run_timeline

def dist_to_anchors(P, anchors):
    return float(np.linalg.norm(P - anchors))

def test_anchor_enables_and_reduces_distance():
    scene = dict(
        frame=0, frame_size=(800, 600),
        labels_init=np.array([[240.,160.],[300.,160.]], float),
        WH=np.array([[60.,24.],[60.,24.]], float),
        labels=[{"anchor_kind":"none"},{"anchor_kind":"none"}],
        points=np.zeros((0,2), float), lines=np.zeros((0,4), float), areas=[],
        anchors=np.array([[180.,150.],[220.,150.]], float),
    )
    cfg = {
        "ll.geom": "rect", "ll.k.repulse": 150.0, "ll.k.inside": 50.0,
        "boundary.k.wall": 80.0, "boundary.wall_eps": 0.3,
        "anchor.k.spring": 10.0,
        "focus.k.attract": 0.0,
        "pl.k.repulse": 0.0, "pl.k.inside": 0.0,
    }
    schedule = [
        {"name": "no_anchor", "scale": {"anchor.k.spring": 0.0}},
        {"name": "with_anchor"},
    ]
    # 跑到第一阶段末
    P1, info1 = run_timeline(scene, cfg, schedule[:1], mode="hybrid", carry_P=True)
    d1 = dist_to_anchors(P1, scene["anchors"])

    # 跑完整两阶段
    P2, info2 = run_timeline(scene, cfg, schedule, mode="hybrid", carry_P=True)
    d2 = dist_to_anchors(P2, scene["anchors"])

    assert d2 < d1, f"distance should decrease after enabling anchor: d1={d1}, d2={d2}"
