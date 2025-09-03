# tests/test_config_completeness.py
import numpy as np
import pytest

from cartoweave.api import solve_frame
from cartoweave.config.utils import (
    merge, numerics, ll, pl, ln, boundary, anchor, focus_huber, report, debug,
    area_embed, area_cross, area_softout,
)

class StrictConfig(dict):
    """
    严格版 cfg：任何缺失键的 .get() 访问都会报错，
    这样可以检测是否存在“函数内部默认值”被偷偷用到。
    """
    def get(self, key, default=None):  # noqa: D401
        if key in self:
            return super().get(key)
        raise AssertionError(f"[STRICT CFG] Missing key accessed: {key!r}")

def make_triangle(cx=400., cy=300., r=120.):
    ang = np.deg2rad(np.array([0., 120., 240.]))
    return np.stack([cx + r*np.cos(ang), cy + r*np.sin(ang)], axis=1).astype(float)

def test_all_params_provided_no_function_defaults_used():
    # --- 场景：每个 term 都会被触发 ---
    poly = make_triangle()
    scene = dict(
        frame=0, frame_size=(800, 600),
        # 两个 label：一个绑定到 area，用来触发 area.embed；另一个不绑定，触发 area.cross
        labels_init=np.array([[380., 320.], [460., 300.]], float),
        WH=np.array([[80., 28.], [80., 28.]], float),
        labels=[{"anchor_kind":"area","anchor_index":0}, {"anchor_kind":"none"}],
        # 点/线/面
        points=np.array([[350., 300.], [600., 500.]], float),                # 触发 pl.rect
        lines=np.array([[300., 260., 520., 260.]], float),                   # 触发 ln.rect
        areas=[{"polygon": poly}],                                           # 触发 area.*
        # 锚点（拉向左右）
        anchors=np.array([[360., 300.], [480., 300.]], float),
    )

    # --- 配置：显式提供所有会被访问的参数（不依赖函数内部默认） ---
    cfg_plain = merge(
        # numerics（公共数值保护）
        numerics(eps_abs=0.5, eps_dist=1e-6, eps_norm=1e-12, beta_softplus_dist=2.0),

        # ll / pl / ln
        ll(geom="rect", k_repulse=150.0, k_inside=50.0, beta_sep=2.0, edge_power=2.0, g_eps=1.0),
        pl(k_repulse=200.0, k_inside=50.0,  beta_sep=2.0, edge_power=2.0, g_eps=1.0),
        ln(k_repulse=180.0, k_inside=80.0,  beta_sep=2.0, edge_power=2.0, g_eps=1.0),

        # boundary
        boundary(k_wall=80.0, wall_power=2.0, wall_eps=0.3, pad=0.0, y_down=True, k_in=0.0),

        # anchor
        anchor(k_spring=10.0, spring_alpha=1.0, r0_points=0.0, r0_lines=0.0),

        # focus（椭圆 pseudo-Huber）
        focus_huber(k_attract=30.0, center=(400., 300.), sigma_x=80.0, sigma_y=120.0, delta=10.0, only_free=False),

        # area.*
        area_embed(k_embed=200.0, k_tan=30.0, ratio_in=0.60, beta_edge=6.0, gate_eta=2.0, gate_slack=1.0),
        area_cross(k_cross=400.0, min_gap=1.5, eta=2.0, alpha=0.35,
                   tan_cap_scale=1.0, use_logcosh=True, sat_p0=2.0, gate_min_interior=0.6),
        area_softout(k_softout=250.0, min_gap=0.0, beta=0.7, alpha=0.35,
                     outside_weight=0.5, out_decay=0.06, in_decay=0.10),
        report(source_topk=0),
        debug(check=False),
    )

    # 用严格字典包一层：任何缺失键访问都会立刻失败
    cfg = StrictConfig(cfg_plain)

    # --- 运行一次完整求解（hybrid：半牛→LBFGS） ---
    # 若某个 term 在实现里用到了 `cfg.get("key", <默认>)` 且我们没提供这个 key，
    # 上面的 StrictConfig 会立刻抛错，测试失败。
    P, info = solve_frame(scene, cfg, mode="hybrid")

    # 走到这里代表没有使用函数内部默认值（全部键都由 cfg 显式提供）
    assert P.shape == scene["labels_init"].shape
