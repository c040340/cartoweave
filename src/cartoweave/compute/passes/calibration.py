from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
import numpy as np

# ========== 工具：读取配置 ==========
def _get_public(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return (cfg.get("compute") or {}).get("public") or {}

def _get_calib_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    comp = (cfg.get("compute") or {})
    return (
        ((comp.get("public") or {}).get("passes") or {}).get("calibration")
        or ((comp.get("passes") or {}).get("calibration") or {})
    )

# ========== 形状档：一次性写入若干“形状/软硬度/ε”等参数（不改 k） ==========
_DEFAULT_PROFILES: Dict[str, Dict[str, float]] = {
    "default": {
        "ll_edge_power": 1.0,
        "boundary_wall_power": 1.0,
        "beta_softplus_dist": 8.0,
        "anchor_spring_alpha": 1.0,
        "area_cross_alpha": 1.0,
        "ll_edge_eps": 1.0,
        "boundary_wall_eps": 1.0,
        "eps_abs": 1e-9,
    },
    "sparse": {
        "ll_edge_power": 0.8,
        "boundary_wall_power": 0.8,
        "beta_softplus_dist": 6.0,
        "anchor_spring_alpha": 0.9,
        "area_cross_alpha": 0.9,
        "ll_edge_eps": 1.0,
        "boundary_wall_eps": 1.0,
        "eps_abs": 1e-9,
    },
    "dense": {
        "ll_edge_power": 1.2,
        "boundary_wall_power": 1.2,
        "beta_softplus_dist": 10.0,
        "anchor_spring_alpha": 1.1,
        "area_cross_alpha": 1.1,
        "ll_edge_eps": 1.0,
        "boundary_wall_eps": 1.0,
        "eps_abs": 1e-9,
    },
}

@dataclass
class ShapeProfilePass:
    """一次性把形状相关参数写入 compute.public.shape_params，不改任何 k。

    挂在 pre-solve/pre-anchor 之前最合适；默认不启用。
    """
    def apply(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        calib = _get_calib_cfg(cfg)
        sp = calib.get("shape_profile") or {}
        if not sp.get("enable", False):
            return cfg

        name = str(sp.get("name", "default"))
        sigma_scale = float(sp.get("sigma_scale", 1.0))
        scope = str(sp.get("scope", "once"))
        public = _get_public(cfg)

        # “已执行标记”避免重复
        memo_key = "_shape_profile_applied"
        if scope == "once" and public.get(memo_key, False):
            return cfg

        prof = _DEFAULT_PROFILES.get(name, _DEFAULT_PROFILES["default"]).copy()
        for k in list(prof.keys()):
            if "sigma" in k:
                prof[k] *= sigma_scale

        shape_params = public.setdefault("shape_params", {})
        shape_params.update(prof)
        public[memo_key] = True
        return cfg

# ========== 多力配平：观测各项力幅值 → 给出 k 的缩放并写回 ==========
def _enabled_terms(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    forces = (_get_public(cfg).get("forces") or {})
    return {n: p for n, p in forces.items() if isinstance(p, dict) and p.get("enable", False)}

def _get_k(cfg: Dict[str, Any], term: str) -> float:
    forces = (_get_public(cfg).get("forces") or {})
    return float(((forces.get(term) or {}).get("k", 0.0)))

def _set_k(cfg: Dict[str, Any], term: str, k_new: float) -> None:
    public = _get_public(cfg)
    forces = public.setdefault("forces", {})
    params = forces.setdefault(term, {})
    params["k"] = float(k_new)

def _winsorized_quantile(x: np.ndarray, p: float, clip_q: float) -> float:
    if x.size == 0:
        return 0.0
    hi = np.quantile(x, clip_q) if x.size > 4 else float(np.max(x))
    x2 = np.clip(x, None, hi)
    return float(np.quantile(x2, p))

@dataclass
class AutoKCalibPass:
    """多力配平：基于观测，把各 term 的 k 拉到目标相对量级。
    注意：需要一个 eval_term_grad 回调；如果你暂时不接线，这个 Pass 放着即可。
    """
    # 由外层注入：term 梯度计算回调  eval_term_grad(term, P, cfg) -> (N,2) ndarray
    eval_term_grad: Optional[Callable[[str, np.ndarray, Dict[str, Any]], Optional[np.ndarray]]] = None

    def apply(self, cfg: Dict[str, Any], P: Optional[np.ndarray] = None) -> Dict[str, Any]:
        calib = _get_calib_cfg(cfg)
        ak = calib.get("auto_k") or {}
        if not ak.get("enable", False):
            return cfg

        scope = str(ak.get("scope", "once"))
        public = _get_public(cfg)
        memo_key = "_auto_k_applied"
        if scope == "once" and public.get(memo_key, False):
            return cfg

        if self.eval_term_grad is None or P is None:
            # 没有接线就什么都不做（保持幂等）
            return cfg

        base_term = str(ak.get("base_term", "focus.attract"))
        p_quantile = float(ak.get("p_quantile", 0.95))
        clip_q = float(ak.get("clip_q", 0.995))
        min_act = float(ak.get("min_activation", 0.05))
        hyst = float(ak.get("hysteresis", 0.25))
        clamp = ak.get("clamp") or {}
        cmin = float(clamp.get("min", 0.33))
        cmax = float(clamp.get("max", 3.0))
        targets = ak.get("targets") or {}

        terms = list(_enabled_terms(cfg).keys())
        # 观测各项幅值
        rep, act = {}, {}
        for name in terms:
            try:
                g = self.eval_term_grad(name, P, cfg)
                if g is None or g.ndim != 2 or g.shape[1] != 2:
                    continue
                m = np.linalg.norm(g, axis=1)
                nz = (m > 0)
                act[name] = float(np.mean(nz)) if m.size > 0 else 0.0
                rep[name] = _winsorized_quantile(m[nz], p_quantile, clip_q) if np.any(nz) else 0.0
            except Exception:
                continue

        # 基准
        base = rep.get(base_term, 0.0)
        if base <= 0:
            vals = [v for (t, v) in rep.items() if act.get(t, 0.0) >= min_act and v > 0]
            base = float(np.median(vals)) if len(vals) else 0.0
        if base <= 0:
            return cfg

        # 建议 → 应用
        for name in terms:
            cur = rep.get(name, 0.0)
            a = act.get(name, 0.0)
            if a < min_act or cur <= 0:
                continue
            tgt_rel = float(targets.get(name, 1.0))
            tgt_abs = tgt_rel * base
            ratio = tgt_abs / cur
            if (1.0 - hyst) <= ratio <= (1.0 + hyst):
                continue
            ratio = max(cmin, min(cmax, ratio))
            _set_k(cfg, name, _get_k(cfg, name) * ratio)

        public[memo_key] = True
        return cfg
