from __future__ import annotations

import numpy as np

from ..primitives.polygons import generate_polygon_by_area
from ..sampling.helpers import frame_metrics

__all__ = ["generate_areas"]

def _sample_trunc_skewnorm(rng: np.random.Generator, mu: float, sigma: float,
                           a: float, b: float, skew: float = 0.0,
                           max_tries: int = 1000) -> float:
    """
    采样一维偏斜正态（Azzalini），并在 [a,b] 内截断；失败则回退到 clamp(mu,a,b)。
    skew=0 退化为标准正态。4σ 约覆盖 [mu-4σ, mu+4σ]。
    """
    if sigma <= 0:
        return float(np.clip(mu, a, b))
    # Azzalini 构造：delta = skew / sqrt(1+skew^2)
    delta = skew / np.sqrt(1.0 + skew * skew) if skew != 0.0 else 0.0
    c = np.sqrt(1.0 - delta * delta)

    for _ in range(max_tries):
        u0 = rng.normal(0.0, 1.0)
        v  = rng.normal(0.0, 1.0)
        # 带偏斜的 N(0,1)
        z = delta * np.abs(u0) + c * v
        x = mu + sigma * z
        if a <= x <= b:
            return float(x)
    # 兜底：取就近截断
    return float(np.clip(mu, a, b))


# --- 替换原来的 generate_areas 函数为下面这个版本 ---
def generate_areas(params, rng: np.random.Generator):
    """
    根据用户配置生成若干多边形：
    1. 目标面积按截断偏斜正态采样（可配置），否则默认占画布面积 2%。
    2. 顶点数随面积占比映射到 [n_vertices_min, n_vertices_max]。
    """
    # --- 画布尺度 ---
    frame_size = (params["frame"]["width"], params["frame"]["height"])
    W, H = frame_size
    diag = float(np.hypot(W, H))
    area_total = float(W * H)

    # --- 读取配置 ---
    cfg = params["random"]["area_gen"]
    n_areas = int(params["counts"]["areas"])
    if n_areas <= 0:
        return []

    area_inset   = float(cfg.get("inset_margin_scale", 0.04)) * diag
    edge_spacing = float(cfg.get("min_edge_spacing_scale", 0.04)) * diag
    n_v_min = int(cfg.get("n_vertices_min", 6))
    n_v_max = int(cfg.get("n_vertices_max", 14))
    n_v_min = max(n_v_min, 3)
    n_v_max = max(n_v_max, n_v_min)

    # 目标面积分布配置（如果没有就用默认）
    ta = cfg.get("target_area", {})
    mu_frac    = float(ta.get("mu_frac",    0.02))
    sigma_frac = float(ta.get("sigma_frac", 0.01))
    skew       = float(ta.get("skew",       0.0))
    min_frac   = float(ta.get("min_frac",   0.005))
    max_frac   = float(ta.get("max_frac",   0.12))
    min_frac = max(1e-6, min(min_frac, 0.95))
    max_frac = max(min_frac, min(max_frac, 0.95))

    polys = []
    for _ in range(n_areas):
        # --- 1) 面积采样：如果 sigma=0，就直接取 mu ---
        if sigma_frac > 0:
            a_frac = _sample_trunc_skewnorm(
                rng=rng,
                mu=mu_frac,
                sigma=sigma_frac,
                a=min_frac,
                b=max_frac,
                skew=skew,
            )
        else:
            a_frac = np.clip(mu_frac, min_frac, max_frac)

        s_target = a_frac * area_total

        # --- 2) 顶点数随面积自适应 ---
        t = np.sqrt((a_frac - min_frac) / (max_frac - min_frac)) if max_frac > min_frac else 0.5
        k = int(np.round(n_v_min + t * (n_v_max - n_v_min)))
        k = int(np.clip(k, n_v_min, n_v_max))

        # --- 3) 调 primitives 造形 ---
        poly = generate_polygon_by_area(
            rng,
            frame_size,
            s_target,
            area_inset,
            edge_spacing,
            (k, k),  # 固定顶点数
        )
        polys.append(poly)

    return polys