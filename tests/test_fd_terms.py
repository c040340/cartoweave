"""
逐项有限差分（FD）测试
---------------------

目的：
- 对每一个力/场 term 单独开启，比较解析梯度 G 与中心差分梯度 G_fd 的最大差异。
- 保证 “能量 E 与梯度 G” 严格配对，且数值误差在可接受范围内。

设计要点：
- 只在被测项的权重(k) > 0 时，该 term 才会启用（依赖 enabled_terms 的门控）。
- softplus / softabs 这类平滑核对差分步长较敏感，选用 H=3e-5。
- 矩形核（ll.rect / pl.rect）在角点附近的光滑近似会带来略大 FD 误差，阈值放宽到 5e-4。
"""

import numpy as np
import copy
import pytest

from cartoweave.compute.eval import energy_and_grad_full

# 各 term 的误差阈值（根据你当前的实测结果设置，留有余量）
TOL = {
    "anchor.spring": 1e-6,  # 接近机器精度
    "boundary.wall": 1e-6,
    "ln.rect":       1e-6,
    "focus.huber":   1e-6,
    "ll.rect":       5e-4,  # 矩形核：softplus/softabs 在角点附近的 FD 误差较大
    "pl.rect":       5e-4,
}

# 有限差分步长：过大会截断，过小会放大舍入误差；对当前核函数这值最稳
H = 3e-5

def fd_err(scene, cfg):
    """
    计算单次场景+配置下的最大 FD 误差：
        max_i | G[i] - G_fd[i] |
    """
    P = scene["labels_init"].copy()
    E, G, comps, _ = energy_and_grad_full(P, scene, np.ones(len(P), bool), cfg)

    flatP = P.reshape(-1)
    G_fd = np.zeros_like(flatP)

    # 中心差分：对每个自由度做一次微小扰动
    for i in range(flatP.size):
        d = np.zeros_like(flatP)
        d[i] = H
        Ep, _, _, _ = energy_and_grad_full((flatP + d).reshape(-1, 2), scene, np.ones(len(P), bool), cfg)
        Em, _, _, _ = energy_and_grad_full((flatP - d).reshape(-1, 2), scene, np.ones(len(P), bool), cfg)
        G_fd[i] = (Ep - Em) / (2 * H)

    return float(np.max(np.abs(G.reshape(-1) - G_fd)))

@pytest.fixture
def base_scene():
    """
    一个极简但包含常见要素的场景：
    - 两个 label（有尺寸 WH）
    - 1 个点、1 条线，便于 pl/ln 的测试
    - anchors 供 anchor.spring 使用
    - areas 暂为空（area.* 在其它测试补）
    """
    return dict(
        frame=0, frame_size=(400, 300),
        labels_init=np.array([[200., 150.], [210., 150.]], float),
        WH=np.array([[60., 24.], [60., 24.]], float),
        anchors=np.array([[180., 150.], [220., 150.]], float),
        points=np.array([[190., 150.]], float),
        lines=np.array([[[100., 140.], [300., 140.]]], float),
        areas=np.zeros((0, 6)),
    )

@pytest.mark.parametrize("name,cfg", [
    ("anchor.spring", {"terms": {"anchor": {"spring": {"k": 10.0}}}}),
    ("boundary.wall", {"boundary.k.wall": 80.0, "boundary.wall_eps": 0.3}),
    ("ln.rect",       {"ln.k.repulse": 180.0, "ln.k.inside": 80.0}),
    ("focus.huber",   {"focus.k.attract": 30.0,
                       "focus.center": np.array([200.,150.]),
                       "focus.sigma.x": 80.0, "focus.sigma.y": 120.0,
                       "focus.delta": 10.0}),
    ("ll.rect",       {"ll.geom": "rect", "ll.k.repulse": 150.0, "ll.k.inside": 50.0}),
    ("pl.rect",       {"pl.k.repulse": 200.0, "pl.k.inside": 50.0}),
])
def test_fd_each_term(base_scene, name, cfg):
    """
    逐项开启测试：
    - 复制 base_scene，防止副作用；
    - cfg 只包含当前被测项的必要参数；
    - 以 max|G - G_fd| 和 TOL[name] 比较。
    """
    sc = copy.deepcopy(base_scene)
    cfg = dict(cfg)  # local copy
    err = fd_err(sc, cfg)
    assert err < TOL[name], f"{name} FD error {err} > {TOL[name]}"
