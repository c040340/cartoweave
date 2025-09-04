"""
Area 系列有限差分测试（embed & cross）
-------------------------------------

说明：
- 仅打开被测项的 k（其他项默认不启用，取决于 enabled_terms 的门控）。
- 误差阈值设置为 5e-4，适配 softabs/softplus 在边附近的数值特性。
- 这里暂不纳入 area.softout 的 FD（等我们最终拍板其能量/衰减形状后再加）。
"""

import numpy as np
import pytest
from cartoweave.engine.core_eval import energy_and_grad_fullP

H = 3e-5
TOL = 5e-4

def fd_err(scene, cfg):
    P = scene["labels_init"].copy()
    E, G, _ = energy_and_grad_fullP(scene, P, cfg)
    flatP = P.reshape(-1)
    G_fd = np.zeros_like(flatP)
    for i in range(flatP.size):
        d = np.zeros_like(flatP); d[i] = H
        Ep, _, _ = energy_and_grad_fullP(scene, (flatP + d).reshape(-1,2), cfg)
        Em, _, _ = energy_and_grad_fullP(scene, (flatP - d).reshape(-1,2), cfg)
        G_fd[i] = (Ep - Em) / (2*H)
    return float(np.max(np.abs(G.reshape(-1) - G_fd)))

def make_triangle(cx=400., cy=300., r=120.):
    ang = np.deg2rad(np.array([0., 120., 240.]))
    return np.stack([cx + r*np.cos(ang), cy + r*np.sin(ang)], axis=1).astype(float)

def test_area_embed_fd():
    # 一个 label 绑定到 area[0]（anchor_kind='area'），驻留在边附近
    poly = make_triangle()
    scene = dict(
        frame=0, frame_size=(800, 600),
        labels_init=np.array([[400., 420.]], float),
        WH=np.array([[80., 28.]], float),
        labels=[{"anchor_kind":"area","anchor_index":0}],
        areas=[{"polygon": poly}],
        points=np.zeros((0,2)), lines=np.zeros((0,2,2)),
        anchors=np.zeros((1,2)),
    )
    cfg = {
        "area.k.embed": 200.0,
        "area.k.tan": 30.0,
        "area.embed.ratio_in": 0.60,
    }
    err = fd_err(scene, cfg)
    assert err < TOL, f"area.embed FD error {err} > {TOL}"

def test_area_cross_fd_logcosh():
    # 两个自由 label，测试与三角形的 cross 作用（log-cosh 模式）
    poly = make_triangle()
    scene = dict(
        frame=0, frame_size=(800, 600),
        labels_init=np.array([[360., 300.],[480., 300.]], float),
        WH=np.array([[80., 28.],[80., 28.]], float),
        labels=[{"anchor_kind":"none"},{"anchor_kind":"none"}],  # 不绑定 area
        areas=[{"polygon": poly}],
        points=np.zeros((0,2)), lines=np.zeros((0,2,2)),
        anchors=np.zeros((2,2)),
    )
    cfg = {
        "area.k.cross": 400.0,
        "area.cross.use_logcosh": True,
        "area.cross.min_gap": 1.5,
    }
    err = fd_err(scene, cfg)
    assert err < TOL, f"area.cross FD error {err} > {TOL}"
