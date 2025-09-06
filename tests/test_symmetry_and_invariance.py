"""
物理对称性与不变性测试
----------------------

包含两个子测试：

1) 作用/反作用（ll.rect）：
   - 仅开启 label↔label term 时，系统内部力应守恒：ΣF ≈ 0。

2) 平移不变性（pl.rect）：
   - 同时平移 labels 与 points，能量 E 应保持不变。

这些测试能在不做有限差分的情况下，快速发现“符号反了 / 漏了一项”等实现错误。
"""

import numpy as np
from cartoweave.compute.eval import energy_and_grad_full
import pytest

def test_ll_rect_action_reaction():
    """
    仅 ll.rect 参与时，内部力守恒（ΣF ≈ 0）：
    - 该性质源于 pairwise 作用在 a/b 两端的力是反向相等的（牛顿第三定律）。
    """
    labels = [{"anchor_kind": "none"} for _ in range(3)]
    scene = dict(
        frame=0,
        frame_size=(1000, 1000),
        labels_init=np.array([[100., 100.], [140., 100.], [120., 130.]], float),
        WH=np.array([[60., 24.], [60., 24.], [60., 24.]], float),
        labels=labels,
        points=np.zeros((0, 2)),
        lines=np.zeros((0, 2, 2)),
        areas=np.zeros((0, 6)),
        anchors=np.zeros((3, 2)),
    )
    cfg = {
        "ll.geom": "rect", "ll.k.repulse": 200.0, "ll.k.inside": 50.0,
        "compute": {"weights": {"ll.rect": 1.0}},
    }

    mask = np.ones(scene["labels_init"].shape[0], bool)
    E, G, comps = energy_and_grad_full(scene["labels_init"], labels, scene, mask, cfg)

    # 仅 ll.rect 时，ΣF ≈ 0
    F = comps.get("ll.rect")
    assert F is not None, "ll.rect not captured; ensure enabled_terms gating by k>0"
    residual = np.abs(F.sum(axis=0)).max()
    assert residual < 1e-8, f"action-reaction violated: resid={residual}"

def test_pl_rect_translation_invariance():
    """
    平移不变性（pl.rect）：
    - 若将 labels 与 points 同时平移同一矢量，pairwise 距离分布不变，能量应不变。
    """
    labels = [{"anchor_kind": "none"} for _ in range(2)]
    scene = dict(
        frame=0,
        frame_size=(10_000, 10_000),
        labels_init=np.array([[300., 300.], [360., 320.]], float),
        WH=np.array([[80., 28.], [80., 28.]], float),
        labels=labels,
        points=np.array([[330., 300.], [450., 350.]], float),
        lines=np.zeros((0, 2, 2)),
        areas=np.zeros((0, 6)),
        anchors=np.zeros((2, 2)),
    )
    cfg = {"pl.k.repulse": 200.0, "pl.k.inside": 50.0}
    cfg.setdefault("compute", {}).setdefault("weights", {})["pl.rect"] = 1.0

    P0 = scene["labels_init"]
    mask0 = np.ones(P0.shape[0], bool)
    E0, _, _ = energy_and_grad_full(P0, labels, scene, mask0, cfg)

    shift = np.array([123.4, -77.7], float)
    scene2 = dict(scene)
    scene2["labels_init"] = P0 + shift
    scene2["points"] = scene["points"] + shift
    mask1 = np.ones(scene2["labels_init"].shape[0], bool)
    E1, _, _ = energy_and_grad_full(scene2["labels_init"], labels, scene2, mask1, cfg)
    assert abs(E0 - E1) < 1e-8, f"translation invariance broken: dE={abs(E0 - E1)}"
