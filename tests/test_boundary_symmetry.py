# tests/test_boundary_symmetry.py
import math
import numpy as np
import pytest

# 这些 import 按你的工程路径写的，如有微调，改成你项目里的实际位置
from cartoweave.config.utils import merge, numerics, boundary
from cartoweave.engine.core_eval import energy_and_grad_fullP


def _make_scene(frame_size=(1080, 1920), w=80.0, h=80.0):
    W, H = frame_size
    return dict(
        frame=0,
        frame_size=(float(W), float(H)),
        # 只放 1 个 label，锚点无关紧要，直接 "none"
        labels=[{"anchor_kind": "none"}],
        labels_init=np.array([[W / 2.0, H / 2.0]], dtype=float),
        WH=np.array([[w, h]], dtype=float),
        points=[], lines=[], areas=[],
    )


def _cfg_only_boundary(k=100.0, p=2.0, eps=0.3, pad=0.0, y_down=True):
    """
    只开启 boundary.wall；其余 term 走默认 0 值。
    如果你在 boundary() 里字段名略有不同（如 k_wall / wall_power / wall_eps / pad），
    请与实际实现对齐。
    """
    return merge(
        numerics(),  # eps_div 等数值兜底
        boundary(
            k_wall=k,
            wall_power=p,
            wall_eps=eps,
            pad=pad,
            y_down=y_down,
            # 把“拉回内侧”的内向项关掉，避免干扰
            k_in=0.0,
        ),
    )


def _probe(scene, cfg, x, y):
    """
    返回 (E, Fx, Fy)；注意 engine 里通常约定 g = -F
    """
    P = np.array([[float(x), float(y)]], dtype=float)
    E, g, comps = energy_and_grad_fullP(scene, P, cfg)
    Fx, Fy = -g[0, 0], -g[0, 1]
    return E, Fx, Fy, comps


def _gaps_at(scene, x, y, pad=0.0):
    """仅用于调试输出，方便定位：四边 gap（以像素为单位）"""
    W, H = scene["frame_size"]
    w, h = scene["WH"][0]
    gl = x - w / 2.0 - pad
    gr = (W - x) - w / 2.0 - pad
    gt = y - h / 2.0 - pad
    gb = (H - y) - h / 2.0 - pad
    return gl, gr, gt, gb


@pytest.mark.parametrize(
    "frame_size", [(1080, 1920), (1920, 1080), (1400, 1400)]
)
def test_boundary_symmetry_square_label(frame_size):
    """
    方形 label（w=h），只开 boundary.wall。
    比较：
      A) 上边中心，距上边 1px
      B) 左边中心，距左边 1px
    预期：E_A ≈ E_B；|F_A| ≈ |F_B|；方向分别沿 +y / +x（若 y_down=True）
    """
    W, H = frame_size
    w = h = 80.0
    pad = 0.0
    cfg = _cfg_only_boundary(k=100.0, p=2.0, eps=0.3, pad=pad, y_down=True)
    scene = _make_scene(frame_size=frame_size, w=w, h=h)

    # A) 上边中心 1px 位置
    xA = W / 2.0
    yA = h / 2.0 + pad + 1.0
    EA, FxA, FyA, compsA = _probe(scene, cfg, xA, yA)

    # B) 左边中心 1px 位置
    xB = w / 2.0 + pad + 1.0
    yB = H / 2.0
    EB, FxB, FyB, compsB = _probe(scene, cfg, xB, yB)

    # 数值对比
    # 用较紧的容差；如实现里用了 softplus 等平滑核，可把 atol 稍放宽
    atol_E = 1e-6
    atol_F = 1e-6

    # 大小应当相等
    assert math.isclose(EA, EB, rel_tol=0, abs_tol=atol_E), (
        f"[boundary symmetry broken: energy]\n"
        f"frame={frame_size}, w=h={w}\n"
        f"E(top@1px)={EA:.9f}, E(left@1px)={EB:.9f}\n"
        f"gaps(top)={_gaps_at(scene, xA, yA, pad)}\n"
        f"gaps(left)={_gaps_at(scene, xB, yB, pad)}\n"
        f"cfg(boundary)={{k_wall:{cfg['boundary']['k_wall']}, p:{cfg['boundary']['wall_power']}, "
        f"eps:{cfg['boundary']['wall_eps']}, pad:{cfg['boundary']['pad']}, y_down:{cfg['boundary']['y_down']}}}"
    )

    FA = (FxA ** 2 + FyA ** 2) ** 0.5
    FB = (FxB ** 2 + FyB ** 2) ** 0.5
    assert math.isclose(FA, FB, rel_tol=0, abs_tol=atol_F), (
        f"[boundary symmetry broken: |F|]\n"
        f"frame={frame_size}, w=h={w}\n"
        f"|F|(top@1px)={FA:.9f} [Fx={FxA:.9f}, Fy={FyA:.9f}]\n"
        f"|F|(left@1px)={FB:.9f} [Fx={FxB:.9f}, Fy={FyB:.9f}]"
    )

    # 方向检查：上边应主要 +y，左边应主要 +x
    assert abs(FyA) > 100 * abs(FxA), (
        f"[direction] top@1px 应主要沿 +y：Fx={FxA:.9e}, Fy={FyA:.9e}"
    )
    assert abs(FxB) > 100 * abs(FyB), (
        f"[direction] left@1px 应主要沿 +x：Fx={FxB:.9e}, Fy={FyB:.9e}"
    )
    assert FyA > 0.0, f"[direction] top@1px Fy 应为正(远离上边)，当前 Fy={FyA:.9e}"
    assert FxB > 0.0, f"[direction] left@1px Fx 应为正(远离左边)，当前 Fx={FxB:.9e}"


@pytest.mark.parametrize(
    "w,h",
    [
        (120.0, 60.0),   # 横向更宽
        (60.0, 120.0),   # 纵向更高
    ],
)
def test_boundary_asymmetry_expected_when_wh_differs(w, h):
    """
    当 w != h 时，“离上 1px”与“离左 1px”本来就不是严格等价——
    因为 gap_x 用 w/2，gap_y 用 h/2。此测试确保我们不要把这种
    “几何非对称”误判为实现 bug：允许它们不同，但检查方向合理，
    同时验证把 label 改回方形能恢复对称（见上一个测试）。
    """
    frame_size = (1080.0, 1920.0)
    W, H = frame_size
    pad = 0.0
    cfg = _cfg_only_boundary(k=100.0, p=2.0, eps=0.3, pad=pad, y_down=True)
    scene = _make_scene(frame_size=frame_size, w=w, h=h)

    xA, yA = W / 2.0, h / 2.0 + pad + 1.0     # 上边 1px
    xB, yB = w / 2.0 + pad + 1.0, H / 2.0     # 左边 1px
    EA, FxA, FyA, _ = _probe(scene, cfg, xA, yA)
    EB, FxB, FyB, _ = _probe(scene, cfg, xB, yB)

    # 方向仍应“远离边界”
    assert FyA > 0.0, f"w={w}, h={h}: top@1px Fy 应为正，现 Fy={FyA:.9e}"
    assert FxB > 0.0, f"w={w}, h={h}: left@1px Fx 应为正，现 Fx={FxB:.9e}"

    # 势能/受力允许不同，不做等价断言；只检查数值有限
    for name, v in [
        ("E_top", EA), ("E_left", EB),
        ("Fx_top", FxA), ("Fy_top", FyA),
        ("Fx_left", FxB), ("Fy_left", FyB),
    ]:
        assert np.isfinite(v), f"{name} 非有限值：{v}"


def test_boundary_1d_profiles_match_for_square_label():
    """
    进一步做一维剖面：只开 boundary，方形 label。
    取 x 方向（y=H/2）与 y 方向（x=W/2）各自从 1..50px 的能量曲线，
    检查它们数列逐点相等（允许极小数值误差）。
    """
    W, H = 1080.0, 1920.0
    w = h = 80.0
    pad = 0.0
    cfg = _cfg_only_boundary(k=37.0, p=2.0, eps=0.25, pad=pad, y_down=True)
    scene = _make_scene(frame_size=(W, H), w=w, h=h)

    xs = np.arange(1, 51, dtype=float)
    # 横向曲线：左边 1..50px
    E_x = []
    for d in xs:
        x = w / 2.0 + pad + d
        y = H / 2.0
        E, *_ = _probe(scene, cfg, x, y)
        E_x.append(E)

    # 纵向曲线：上边 1..50px
    E_y = []
    for d in xs:
        x = W / 2.0
        y = h / 2.0 + pad + d
        E, *_ = _probe(scene, cfg, x, y)
        E_y.append(E)

    E_x = np.asarray(E_x)
    E_y = np.asarray(E_y)
    # 逐点比较（相同 d）
    assert np.allclose(E_x, E_y, rtol=0.0, atol=1e-6), (
        "[boundary 1D profiles mismatch]\n"
        f"E_x[:5]={E_x[:5]}\nE_y[:5]={E_y[:5]}"
    )
