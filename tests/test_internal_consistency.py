"""
内部一致性测试：ΣF + ∇E ≈ 0
----------------------------

目的：
- 在 F = -∇E 的全局约定下，“所有 term 的分力之和 + 解析梯度”应接近 0。
- 该测试通过 record 钩子捕获每个 term 的分力，并进行合力一致性校验。

备注：
- 该用例同时开启多个常见项（anchor、ll.rect、boundary），检查组合情况下的一致性。
"""

import numpy as np
from cartoweave.engine.core_eval import energy_and_grad_fullP

def test_sumF_plus_grad_zero_anchor_ll_boundary():
    # 构造一个小场景；WH/anchors 的数量匹配 labels
    scene = dict(
        frame=0, frame_size=(400, 300),
        labels_init=np.array([[200., 150.], [212., 152.]], float),
        WH=np.array([[60., 24.], [60., 24.]], float),
        anchors=np.array([[180., 150.], [220., 150.]], float),
        points=np.zeros((0, 2)), lines=np.zeros((0, 2, 2)), areas=np.zeros((0, 6)),
    )
    cfg = {
        "terms": {"anchor": {"spring": {"k": 10.0}}},
        "ll.geom": "rect", "ll.k.repulse": 150.0, "ll.k.inside": 50.0,
        "boundary.k.wall": 80.0, "boundary.wall_eps": 0.3,
    }

    # 通过 record 收集每个 term 的分力
    comps_box = {}
    def rec(P, E, comps, sources):
        comps_box.clear()
        # 深拷贝到本地，避免外部后续修改
        comps_box.update({k: v.copy() for k, v in comps.items()})

    # 计算 E 与 G，同时填充 comps_box
    E, G, _ = energy_and_grad_fullP(scene, scene["labels_init"], cfg, record=rec)

    # 将所有分力相加，按 F = -∇E 约定应满足 ΣF + G ≈ 0
    Fsum = 0.0 * G
    for V in comps_box.values():
        Fsum += V
    resid = float(np.abs(Fsum + G).max())
    assert resid < 1e-6, f"sum(F) + grad ≠ 0, resid={resid}"
