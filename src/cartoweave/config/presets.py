# src/cartoweave/config/presets.py
from __future__ import annotations
from typing import Dict, Any
from .utils import (
    merge, validate,
    numerics, ll, pl, ln, boundary, anchor, focus_huber, report,
    area_embed, area_cross, area_softout,
)


# ---------------------------------------------------------------------------
# Visualisation configuration toggles
# ---------------------------------------------------------------------------

# ``VIZ_FORCE_CONFIG`` centralises thresholds that affect how forces are
# displayed in the viewer.  The info panel always lists all force terms while
# the arrow plot may hide tiny components for clarity.  Consumers import this
# dictionary and read or update the values directly.
VIZ_FORCE_CONFIG: Dict[str, Any] = {
    # Info panel: show every force term without filtering.
    "info_show_all_terms": True,

    # Arrow plot: hide small arrows purely on the UI side.  These thresholds do
    # not affect the recorded force data or the info panel above.
    "arrows_min_abs": 1e-9,
    "arrows_min_ratio": 1e-2,
    "arrows_max_terms": None,  # Optional cap on the number of terms drawn.
    "arrows_show_all": False,  # Developer override to draw all terms.
}

def minimal_cfg() -> Dict[str, Any]:
    """
    所有 k=0 的最小模板，仅提供数值保护参数。
    """
    cfg = merge(
        numerics(),  # eps/beta 等
        # 其它分段按需添加
    )
    validate(cfg)
    return cfg

def default_cfg() -> Dict[str, Any]:
    """
    一个开箱即用的默认配方：
    - ll/pl/ln + boundary + anchor + focus(huber)
    - area* 默认关闭
    """
    cfg = merge(
        numerics(eps_abs=0.5, eps_dist=1e-6, eps_norm=1e-12, beta_softplus_dist=2.0),
        ll(geom="rect", k_repulse=150.0, k_inside=50.0, beta_sep=2.0, beta_in=6.0, edge_power=2.0, g_eps=1.0),
        pl(k_repulse=200.0, k_inside=50.0, beta_sep=2.0, edge_power=2.0, g_eps=1.0),
        ln(k_repulse=180.0, k_inside=80.0,  beta_sep=2.0, beta_in=6.0, edge_power=2.0, g_eps=1.0),
        boundary(k_wall=80.0, wall_eps=0.3),
        anchor(k_spring=10.0),
        focus_huber(k_attract=30.0, center=(400.0, 300.0), sigma_x=80.0, sigma_y=120.0, delta=10.0),
        report(source_topk=0),
        area_embed(k_embed=200.0, k_tan=30.0, ratio_in=0.60),
        area_cross(k_cross=400.0),
        area_softout(k_softout=250.0),
    )
    validate(cfg)
    return cfg

def focus_only_cfg() -> Dict[str, Any]:
    """
    只打开 focus 椭圆 Huber（方便做方向/集聚实验）。
    """
    cfg = merge(
        numerics(),
        focus_huber(k_attract=40.0, center=(400.0, 300.0), sigma_x=100.0, sigma_y=100.0, delta=8.0),
    )
    validate(cfg)
    return cfg

def area_pack_cfg() -> Dict[str, Any]:
    """
    面相关三件套的示例（按需叠加到 default_cfg 上）。
    """
    cfg = merge(
        area_embed(k_embed=200.0, k_tan=30.0, ratio_in=0.60, beta_edge=6.0),
        area_cross(k_cross=400.0, min_gap=1.5, use_logcosh=True),
        area_softout(k_softout=250.0, min_gap=0.0),
    )
    validate(cfg)
    return cfg
