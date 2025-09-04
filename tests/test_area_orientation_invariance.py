# tests/test_area_orientation_invariance.py
import numpy as np
from cartoweave.engine.core_eval import energy_and_grad_fullP

def make_triangle(cx=400., cy=300., r=120.):
    ang = np.deg2rad(np.array([0., 120., 240.]))
    poly = np.stack([cx + r*np.cos(ang), cy + r*np.sin(ang)], axis=1).astype(float)
    return poly

def test_embed_orientation_invariance():
    poly_ccw = make_triangle()
    poly_cw  = poly_ccw[::-1].copy()

    scene_ccw = dict(
        frame=0, frame_size=(800, 600),
        labels_init=np.array([[400., 420.]], float),
        WH=np.array([[80., 28.]], float),
        labels=[{"anchor_kind":"area","anchor_index":0}],
        areas=[{"polygon": poly_ccw}],
        points=np.zeros((0,2)), lines=np.zeros((0,2,2)),
        anchors=np.zeros((1,2)),
    )
    scene_cw = dict(scene_ccw)
    scene_cw["areas"] = [{"polygon": poly_cw}]

    cfg = {"area.k.embed": 200.0, "area.k.tan": 30.0, "area.embed.ratio_in": 0.60}

    E1, G1, _ = energy_and_grad_fullP(scene_ccw, scene_ccw["labels_init"], cfg)
    E2, G2, _ = energy_and_grad_fullP(scene_cw,  scene_cw["labels_init"],  cfg)

    assert abs(E1 - E2) < 1e-8
    assert float(np.max(np.abs(G1 - G2))) < 1e-6

def test_cross_orientation_invariance():
    poly_ccw = make_triangle()
    poly_cw  = poly_ccw[::-1].copy()

    scene_ccw = dict(
        frame=0, frame_size=(800, 600),
        labels_init=np.array([[360., 300.],[480., 300.]], float),
        WH=np.array([[80., 28.],[80., 28.]], float),
        labels=[{"anchor_kind":"none"},{"anchor_kind":"none"}],
        areas=[{"polygon": poly_ccw}],
        points=np.zeros((0,2)), lines=np.zeros((0,2,2)),
        anchors=np.zeros((2,2)),
    )
    scene_cw = dict(scene_ccw)
    scene_cw["areas"] = [{"polygon": poly_cw}]

    cfg = {"area.k.cross": 400.0, "area.cross.use_logcosh": True, "area.cross.min_gap": 1.5}

    E1, G1, _ = energy_and_grad_fullP(scene_ccw, scene_ccw["labels_init"], cfg)
    E2, G2, _ = energy_and_grad_fullP(scene_cw,  scene_cw["labels_init"],  cfg)

    assert abs(E1 - E2) < 1e-8
    assert float(np.max(np.abs(G1 - G2))) < 1e-6
