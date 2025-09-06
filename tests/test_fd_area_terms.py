import numpy as np
import pytest
from cartoweave.compute.eval import energy_and_grad_full

H = 3e-5
TOL = 5e-4


def make_triangle(cx=400.0, cy=300.0, r=120.0):
    ang = np.deg2rad(np.array([0.0, 120.0, 240.0]))
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1).astype(float)


def fd_err(P0, labels, scene, active_mask, cfg):
    E, G, comps = energy_and_grad_full(P0, labels, scene, active_mask, cfg)
    flatP = P0.reshape(-1)
    G_fd = np.zeros_like(flatP)
    for i in range(flatP.size):
        d = np.zeros_like(flatP)
        d[i] = H
        Ep, _, _ = energy_and_grad_full((flatP + d).reshape(-1, 2), labels, scene, active_mask, cfg)
        Em, _, _ = energy_and_grad_full((flatP - d).reshape(-1, 2), labels, scene, active_mask, cfg)
        G_fd[i] = (Ep - Em) / (2 * H)
    return float(np.max(np.abs(G.reshape(-1) - G_fd)))


def test_area_embed_fd():
    P0 = np.array([[400.0, 420.0]], float)
    labels = [{"anchor_kind": "area", "anchor_index": 0}]
    scene = dict(
        frame=0,
        frame_size=(800, 600),
        labels_init=P0,
        WH=np.array([[80.0, 28.0]], float),
        labels=labels,
        areas=[{"polygon": make_triangle()}],
        points=np.zeros((0, 2)),
        lines=np.zeros((0, 2, 2)),
    )
    active = np.ones(1, bool)
    cfg = {"terms": {"area_embed": {"k": 200.0, "sigma": 6.0}}}
    err = fd_err(P0, labels, scene, active, cfg)
    assert np.isfinite(err)


def test_area_cross_fd_logcosh():
    P0 = np.array([[360.0, 300.0], [480.0, 300.0]], float)
    labels = [{"anchor_kind": "none"}, {"anchor_kind": "none"}]
    scene = dict(
        frame=0,
        frame_size=(800, 600),
        labels_init=P0,
        WH=np.array([[80.0, 28.0], [80.0, 28.0]], float),
        labels=labels,
        areas=[{"polygon": make_triangle()}],
        points=np.zeros((0, 2)),
        lines=np.zeros((0, 2, 2)),
    )
    active = np.ones(2, bool)
    cfg = {
        "terms": {"area_cross": {"k": 400.0, "sigma": 6.0}},
        "area.cross.use_logcosh": True,
        "area.cross.min_gap": 1.5,
    }
    err = fd_err(P0, labels, scene, active, cfg)
    assert err < TOL, f"area.cross FD error {err} > {TOL}"
