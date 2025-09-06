# tests/test_area_orientation_invariance.py
import numpy as np
from cartoweave.compute.eval import energy_and_grad_full
import pytest

def make_triangle(cx=400., cy=300., r=120.):
    ang = np.deg2rad(np.array([0., 120., 240.]))
    poly = np.stack([cx + r*np.cos(ang), cy + r*np.sin(ang)], axis=1).astype(float)
    return poly

def test_embed_orientation_invariance():
    poly_ccw = make_triangle()
    poly_cw  = poly_ccw[::-1].copy()

    labels_ccw = [{"anchor_kind": "area", "anchor_index": 0}]
    scene_ccw = dict(
        frame=0,
        frame_size=(800, 600),
        labels_init=np.array([[400., 420.]], float),
        WH=np.array([[80., 28.]], float),
        labels=labels_ccw,
        areas=[{"polygon": poly_ccw}],
        points=np.zeros((0, 2)),
        lines=np.zeros((0, 2, 2)),
        anchors=np.zeros((1, 2)),
    )
    scene_cw = dict(scene_ccw)
    scene_cw["areas"] = [{"polygon": poly_cw}]

    cfg = {"area.k.embed": 200.0, "area.k.tan": 30.0, "area.embed.ratio_in": 0.60}

    mask_ccw = np.ones(scene_ccw["labels_init"].shape[0], bool)
    mask_cw = np.ones(scene_cw["labels_init"].shape[0], bool)
    E1, G1, comps1 = energy_and_grad_full(scene_ccw["labels_init"], labels_ccw, scene_ccw, mask_ccw, cfg)
    E2, G2, comps2 = energy_and_grad_full(scene_cw["labels_init"], labels_ccw, scene_cw, mask_cw, cfg)

    assert abs(E1 - E2) < 1e-8
    assert float(np.max(np.abs(G1 - G2))) < 1e-6

def test_cross_orientation_invariance():
    poly_ccw = make_triangle()
    poly_cw  = poly_ccw[::-1].copy()

    labels_ccw = [{"anchor_kind": "none"}, {"anchor_kind": "none"}]
    scene_ccw = dict(
        frame=0,
        frame_size=(800, 600),
        labels_init=np.array([[360., 300.], [480., 300.]], float),
        WH=np.array([[80., 28.], [80., 28.]], float),
        labels=labels_ccw,
        areas=[{"polygon": poly_ccw}],
        points=np.zeros((0, 2)),
        lines=np.zeros((0, 2, 2)),
        anchors=np.zeros((2, 2)),
    )
    scene_cw = dict(scene_ccw)
    scene_cw["areas"] = [{"polygon": poly_cw}]

    cfg = {"area.k.cross": 400.0, "area.cross.use_logcosh": True, "area.cross.min_gap": 1.5}

    mask_ccw = np.ones(scene_ccw["labels_init"].shape[0], bool)
    mask_cw = np.ones(scene_cw["labels_init"].shape[0], bool)
    E1, G1, comps1 = energy_and_grad_full(scene_ccw["labels_init"], labels_ccw, scene_ccw, mask_ccw, cfg)
    E2, G2, comps2 = energy_and_grad_full(scene_cw["labels_init"], labels_ccw, scene_cw, mask_cw, cfg)

    assert abs(E1 - E2) < 1e-8
    assert float(np.max(np.abs(G1 - G2))) < 1e-6
