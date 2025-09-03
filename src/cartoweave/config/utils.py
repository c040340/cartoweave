# src/cartoweave/config/utils.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

# ---------------------------
# 基础：合并 & 校验
# ---------------------------

def merge(*cfgs: Dict[str, Any]) -> Dict[str, Any]:
    """
    从左到右合并多个 dict（右侧覆盖左侧）。
    仅做浅合并；值都是标量或 ndarray。
    """
    out: Dict[str, Any] = {}
    for c in cfgs:
        if c:
            out.update(c)
    return out

# 允许的键（新仓统一点分命名）
_ALLOWED_KEYS = {
    # numerics
    "eps.abs", "eps.dist", "eps.norm", "beta.softplus.dist",
    # ll/pl/ln
    "ll.geom", "ll.k.repulse", "ll.k.inside", "ll.beta.sep", "ll.beta.in", "ll.edge_power", "ll.g_eps", "ll.center_eps", "ll.disk.mode",
    "pl.k.repulse", "pl.k.inside", "pl.beta.sep", "pl.beta.in", "pl.edge_power", "pl.g_eps",
    "ln.k.repulse", "ln.k.inside", "ln.beta.sep", "ln.beta.in", "ln.edge_power", "ln.g_eps",
    # boundary
    "boundary.k.wall", "boundary.wall_power", "boundary.wall_eps", "boundary.pad", "boundary.y_down", "boundary.k.in",
    # anchor
    "anchor.k.spring", "anchor.spring.alpha", "anchor.r0.points", "anchor.r0.lines",
    # focus（椭圆 pseudo-Huber）
    "focus.k.attract", "focus.center", "focus.sigma.x", "focus.sigma.y", "focus.delta", "focus.only_free",
    # area.*
    "area.k.embed", "area.k.tan", "area.embed.ratio_in", "area.embed.beta_edge", "area.tan.gate.eta", "area.tan.gate.slack",
    "area.k.cross", "area.cross.min_gap", "area.cross.eta", "area.cross.alpha", "area.cross.tan_cap_scale",
    "area.cross.use_logcosh", "area.cross.sat_p0", "area.cross.gate_min_interior",
    "area.k.softout", "area.softout.min_gap", "area.softout.beta", "area.softout.alpha",
    "area.softout.outside_weight", "area.softout.out_decay", "area.softout.in_decay",
    # reporting / debug
    "debug.check",
    "source.topk",
    # visualization
    "viz.show", "viz.field.kind", "viz.field.cmap",
}

def report(source_topk: int = 0) -> Dict[str, Any]:
    """信息记录相关参数，目前只用到 source.topk（0 表示不截断）"""
    return {"source.topk": int(source_topk)}

def debug(check: bool = False) -> Dict[str, Any]:
    """调试开关：True 会在 core_eval 等处走额外检查/日志"""
    return {"debug.check": bool(check)}

def validate(cfg: Dict[str, Any]) -> None:
    """
    轻量校验：
      - 只允许 _ALLOWED_KEYS 中的键
      - focus.center 自动规范为 np.array([cx, cy], float)
      - 数值型键需是 int/float（或 bool），数组键需是 1x2 的 array-like
    发现未知键会抛 ValueError。
    """
    unknown = [k for k in cfg.keys() if k not in _ALLOWED_KEYS]
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    # 统一 focus.center
    if "focus.center" in cfg:
        v = cfg["focus.center"]
        arr = np.asarray(v, dtype=float).reshape(2,)
        cfg["focus.center"] = arr

def enabled_terms(cfg: Dict[str, Any]) -> Dict[str, bool]:
    """
    按 “k>0” 判断各 term 是否启用（仅用于查看/调参，不影响引擎）。
    """
    return {
        "ll":        cfg.get("ll.k.repulse", 0) > 0 or cfg.get("ll.k.inside", 0) > 0,
        "pl":        cfg.get("pl.k.repulse", 0) > 0 or cfg.get("pl.k.inside", 0) > 0,
        "ln":        cfg.get("ln.k.repulse", 0) > 0 or cfg.get("ln.k.inside", 0) > 0,
        "boundary":  cfg.get("boundary.k.wall", 0) > 0 or cfg.get("boundary.k.in", 0) > 0,
        "anchor":    cfg.get("anchor.k.spring", 0) > 0,
        "focus":     cfg.get("focus.k.attract", 0) > 0,
        "area.embed":   cfg.get("area.k.embed", 0) > 0 or cfg.get("area.k.tan", 0) > 0,
        "area.cross":   cfg.get("area.k.cross", 0) > 0,
        "area.softout": cfg.get("area.k.softout", 0) > 0,
    }

# ---------------------------
# 分段 builder（纯 dict）
# ---------------------------

def numerics(
    eps_abs: float = 0.5,
    eps_dist: float = 1e-6,
    eps_norm: float = 1e-12,
    beta_softplus_dist: float = 2.0,
) -> Dict[str, Any]:
    return {
        "eps.abs": eps_abs,
        "eps.dist": eps_dist,
        "eps.norm": eps_norm,
        "beta.softplus.dist": beta_softplus_dist,
    }

def ll(
    geom: str = "rect",
    k_repulse: float = 0.0,
    k_inside: float = 0.0,
    beta_sep: float = 2.0,
    beta_in: float = 6.0,
    edge_power: float = 2.0,
    g_eps: float = 1.0,
    center_eps: float = 0.5,
    disk_mode: str = "max",  # 仅当 geom="disk" 时有意义
) -> Dict[str, Any]:
    return {
        "ll.geom": geom,
        "ll.k.repulse": k_repulse,
        "ll.k.inside": k_inside,
        "ll.beta.sep": beta_sep,
        "ll.beta.in": beta_in,
        "ll.edge_power": edge_power,
        "ll.g_eps": g_eps,
        "ll.center_eps": center_eps,
        "ll.disk.mode": disk_mode,
    }

def pl(
    k_repulse: float = 0.0,
    k_inside: float = 0.0,
    beta_sep: float = 2.0,
    beta_in: float = 6.0,
    edge_power: float = 2.0,
    g_eps: float = 1.0,
) -> Dict[str, Any]:
    return {
        "pl.k.repulse": k_repulse,
        "pl.k.inside": k_inside,
        "pl.beta.sep": beta_sep,
        "pl.beta.in": beta_in,
        "pl.edge_power": edge_power,
        "pl.g_eps": g_eps,
    }

def ln(
    k_repulse: float = 0.0,
    k_inside: float = 0.0,
    beta_sep: float = 2.0,
    beta_in: float = 6.0,
    edge_power: float = 2.0,
    g_eps: float = 1.0,
) -> Dict[str, Any]:
    return {
        "ln.k.repulse": k_repulse,
        "ln.k.inside": k_inside,
        "ln.beta.sep": beta_sep,
        "ln.beta.in": beta_in,
        "ln.edge_power": edge_power,
        "ln.g_eps": g_eps,
    }

def boundary(
    k_wall: float = 0.0,
    wall_power: float = 2.0,
    wall_eps: float = 0.3,
    pad: float = 0.0,
    y_down: bool = True,
    k_in: float = 0.0,
) -> Dict[str, Any]:
    return {
        "boundary.k.wall": k_wall,
        "boundary.wall_power": wall_power,
        "boundary.wall_eps": wall_eps,
        "boundary.pad": pad,
        "boundary.y_down": y_down,
        "boundary.k.in": k_in,
    }

def anchor(
    k_spring: float = 0.0,
    spring_alpha: float = 1.0,
    r0_points: float = 0.0,
    r0_lines: float = 0.0,
) -> Dict[str, Any]:
    return {
        "anchor.k.spring": k_spring,
        "anchor.spring.alpha": spring_alpha,
        "anchor.r0.points": r0_points,
        "anchor.r0.lines": r0_lines,
    }

def focus_huber(
    k_attract: float = 0.0,
    center: Tuple[float, float] | None = None,
    sigma_x: float = 100.0,
    sigma_y: float = 100.0,
    delta: float = 8.0,
    only_free: bool = False,
) -> Dict[str, Any]:
    cfg = {
        "focus.k.attract": k_attract,
        "focus.sigma.x": sigma_x,
        "focus.sigma.y": sigma_y,
        "focus.delta": delta,
        "focus.only_free": only_free,
    }
    if center is not None:
        cfg["focus.center"] = np.asarray(center, dtype=float).reshape(2,)
    return cfg

def area_embed(
    k_embed: float = 0.0,
    k_tan: float = 0.0,
    ratio_in: float = 0.60,
    beta_edge: float = 6.0,
    gate_eta: float = 2.0,
    gate_slack: float = 1.0,
) -> Dict[str, Any]:
    return {
        "area.k.embed": k_embed,
        "area.k.tan": k_tan,
        "area.embed.ratio_in": ratio_in,
        "area.embed.beta_edge": beta_edge,
        "area.tan.gate.eta": gate_eta,
        "area.tan.gate.slack": gate_slack,
    }

def area_cross(
    k_cross: float = 0.0,
    min_gap: float = 1.5,
    eta: float = 2.0,
    alpha: float = 0.35,
    tan_cap_scale: float = 1.0,
    use_logcosh: bool = True,
    sat_p0: float = 2.0,
    gate_min_interior: float = 0.6,
) -> Dict[str, Any]:
    return {
        "area.k.cross": k_cross,
        "area.cross.min_gap": min_gap,
        "area.cross.eta": eta,
        "area.cross.alpha": alpha,
        "area.cross.tan_cap_scale": tan_cap_scale,
        "area.cross.use_logcosh": use_logcosh,
        "area.cross.sat_p0": sat_p0,
        "area.cross.gate_min_interior": gate_min_interior,
    }

def area_softout(
    k_softout: float = 0.0,
    min_gap: float = 0.0,
    beta: float = 0.7,
    alpha: float = 0.35,
    outside_weight: float = 0.5,
    out_decay: float = 0.06,
    in_decay: float = 0.10,
) -> Dict[str, Any]:
    return {
        "area.k.softout": k_softout,
        "area.softout.min_gap": min_gap,
        "area.softout.beta": beta,
        "area.softout.alpha": alpha,
        "area.softout.outside_weight": outside_weight,
        "area.softout.out_decay": out_decay,
        "area.softout.in_decay": in_decay,
    }


def viz(show: bool = False, field_kind: str = "3d", field_cmap: str = "viridis") -> Dict[str, Any]:
    """Visualization options exposed via config.

    Parameters
    ----------
    show:
        Whether to launch the interactive viewer after solving.
    field_kind:
        ``"3d"`` for surface plot or ``"heatmap"`` for 2D visualization.
    field_cmap:
        Matplotlib colormap name used when rendering scalar fields.
    """
    return {
        "viz.show": bool(show),
        "viz.field.kind": str(field_kind),
        "viz.field.cmap": str(field_cmap),
    }
