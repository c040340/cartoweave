# tests/test_boundary_locate_asymmetry.py
import math
import numpy as np
import pytest

from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.utils import kernels

def _scene(frame_size=(1080, 1920), w=80.0, h=80.0):
    W, H = frame_size
    return dict(
        frame=0,
        frame_size=(float(W), float(H)),
        labels=[{"anchor_kind": "none"}],
        labels_init=np.array([[W/2.0, H/2.0]], float),
        WH=np.array([[w, h]], float),
        points=[], lines=[], areas=[],
    )

def _cfg_boundary(k=100.0, p=2.0, eps=0.3, pad=0.0, y_down=True):
    # 按你的项目：平铺式 key
    return {
        "boundary.k.wall": float(k),
        "boundary.wall_power": float(p),
        "boundary.wall_eps": float(eps),
        "boundary.pad": float(pad),
        "boundary.k.in": 0.0,        # ≤0 触发 auto-match；我们保持与实现一致
        "boundary.y_down": bool(y_down),

        # 禁掉其他项，纯看 boundary
        "ll.k.repulse": 0.0, "ll.k.inside": 0.0,
        "pl.k.repulse": 0.0, "pl.k.inside": 0.0,
        "ln.k.repulse": 0.0, "ln.k.inside": 0.0,
        "focus.k.huber": 0.0, "focus.k.quad": 0.0,
        "area.k.embed": 0.0, "area.k.cross": 0.0, "area.k.softout": 0.0,
        "beta.softplus.dist": 3.0,
    }

def _piece_energy(s, k_wall, power, beta_d, eps_div, k_in):
    # 与 term_boundary.py::piece 完全对齐（能量 + 沿法向的标量力幅）
    c  = kernels.softplus(s,  beta_d) + eps_div
    v  = kernels.softplus(-s, beta_d)
    sc = kernels.sigmoid(beta_d * s)
    sv = kernels.sigmoid(-beta_d * s)
    E_out  = kernels.invdist_energy(c, k_wall, power)
    F_outm = kernels.invdist_force_mag(c, k_wall, power) * sc
    E_in   = 0.5 * k_in * (v * v)
    F_inm  = k_in * v * sv
    fm = F_outm + F_inm
    return (E_out + E_in), fm

def _decompose_per_side(scene, cfg, x, y):
    W, H = scene["frame_size"]
    w, h = scene["WH"][0]
    k_wall = cfg["boundary.k.wall"]
    power  = cfg["boundary.wall_power"]
    pad    = cfg["boundary.pad"]
    eps_div= cfg["boundary.wall_eps"]
    beta_d = cfg["beta.softplus.dist"]
    # auto-match inside 系数（和实现一致）
    v0 = math.log(2.0) / max(beta_d, 1e-8)
    e0 = v0 + eps_div
    k_in = k_wall / ((e0 ** power) * max(v0, 1e-8)) if cfg.get("boundary.k.in", 0.0) <= 0.0 else cfg["boundary.k.in"]

    cx, cy = float(x), float(y)
    hx, hy = 0.5 * w, 0.5 * h

    sL = (cx - hx) - pad
    sR = (W - pad) - (cx + hx)
    sT = (cy - hy) - pad
    sB = (H - pad) - (cy + hy)

    EL, _ = _piece_energy(sL, k_wall, power, beta_d, eps_div, k_in)
    ER, _ = _piece_energy(sR, k_wall, power, beta_d, eps_div, k_in)
    ET, _ = _piece_energy(sT, k_wall, power, beta_d, eps_div, k_in)
    EB, _ = _piece_energy(sB, k_wall, power, beta_d, eps_div, k_in)

    return dict(E=dict(L=EL, R=ER, T=ET, B=EB), s=dict(L=sL, R=sR, T=sT, B=sB))

@pytest.mark.parametrize("frame_size", [(1080, 1920), (1920, 1080), (1400, 1400)])
def test_boundary_asymmetry_is_from_far_sides(frame_size, capsys):
    """
    诊断：上边 1px vs 左边 1px。
    断言 1：近侧边能量应当一致（ET(top) == EL(left)）。
    断言 2：两处总能量差 == 其余三边能量和的差 —— 这直接定位到“远侧三边”。
    并在 -s 模式下输出逐边分解表。
    """
    W, H = frame_size
    sc = _scene(frame_size=frame_size, w=80.0, h=80.0)
    cfg = _cfg_boundary(k=37.0, p=2.0, eps=0.25, pad=0.0, y_down=True)

    # 对称探针
    x_top,  y_top  = W/2.0, 80.0/2.0 + 1.0
    x_left, y_left = 80.0/2.0 + 1.0, H/2.0

    mask = np.array([True], bool)
    E_top, *_  = energy_and_grad_full(np.array([[x_top,  y_top]], float), sc, mask, cfg)
    E_left, *_ = energy_and_grad_full(np.array([[x_left, y_left]], float), sc, mask, cfg)

    top  = _decompose_per_side(sc, cfg, x_top,  y_top)
    left = _decompose_per_side(sc, cfg, x_left, y_left)

    # 1) 近侧边能量一致
    assert math.isclose(top["E"]["T"], left["E"]["L"], rel_tol=0.0, abs_tol=1e-8), \
        f"Near-side energies differ: ET(top)={top['E']['T']} vs EL(left)={left['E']['L']}"

    # 2) 差值由远侧三边导致
    far_top  = top["E"]["L"] + top["E"]["R"] + top["E"]["B"]
    far_left = left["E"]["T"] + left["E"]["B"] + left["E"]["R"]
    diff_E   = E_top - E_left
    diff_far = far_top - far_left
    assert math.isclose(diff_E, diff_far, rel_tol=0.0, abs_tol=1e-8), \
        f"diff_E={diff_E} not equal to far-sides delta={diff_far}"

    # 3) 输出报告（pytest -s 可见）
    print(f"\n[Boundary diagnostic @ frame={frame_size}]")
    print(f"  E(top@1px)  = {E_top:.9f}")
    print(f"  E(left@1px) = {E_left:.9f}")
    print("  Per-side energies (L,R,T,B):")
    print(f"    top : {top['E']['L']:.9f}, {top['E']['R']:.9f}, {top['E']['T']:.9f}, {top['E']['B']:.9f}")
    print(f"    left: {left['E']['L']:.9f}, {left['E']['R']:.9f}, {left['E']['T']:.9f}, {left['E']['B']:.9f}")
    print(f"  diff_E={diff_E:.9f} | far_top={far_top:.9f}, far_left={far_left:.9f}, far_delta={diff_far:.9f}")

    # 4) 在正方形画幅下（W==H），应当恢复“几乎相等”
    if W == H:
        assert math.isclose(E_top, E_left, rel_tol=0.0, abs_tol=1e-6), \
            f"In a square frame {frame_size}, energies should match but got {E_top} vs {E_left}"

def test_boundary_profile_x_vs_y_equal_only_when_frame_is_square():
    """
    x 向（左边 1..50px）与 y 向（上边 1..50px）的一维能量曲线。
    在 W==H 下两条曲线应逐点相等；在 W!=H 下，两条曲线相差一个几乎常量的偏移，
    偏移来自“远侧三边”的不同，且对 d（1..50px）不敏感。
    """
    for frame_size in [(1080, 1920), (1400, 1400)]:
        W, H = frame_size
        sc = _scene(frame_size=frame_size, w=80.0, h=80.0)
        cfg = _cfg_boundary(k=37.0, p=2.0, eps=0.25, pad=0.0, y_down=True)
        mask = np.array([True], bool)

        xs = np.arange(1.0, 51.0, 1.0)
        E_x, E_y = [], []
        for d in xs:
            E,_g,_,_ = energy_and_grad_full(np.array([[80.0/2.0 + d, H/2.0]], float), sc, mask, cfg)
            E_x.append(E)
            E,_g,_,_ = energy_and_grad_full(np.array([[W/2.0, 80.0/2.0 + d]], float), sc, mask, cfg)
            E_y.append(E)
        E_x = np.asarray(E_x); E_y = np.asarray(E_y)
        if W == H:
            assert np.allclose(E_x, E_y, rtol=0.0, atol=1e-6)
        else:
            delta = E_y - E_x
            assert np.all(delta > 0) or np.all(delta < 0), \
                "Expected a consistent sign difference in non-square frame"
            assert np.std(delta) < 1e-3, f"Offset not nearly constant: std={np.std(delta)}"
