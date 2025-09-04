# src/cartoweave/engine/forces/__init__.py
from __future__ import annotations
from typing import Dict, Callable, Any, List

# 所有 term 的注册表；每个 term 文件里用 @register("name") 装饰器登记
REGISTRY: Dict[str, Callable] = {}
_REGISTRY_BASE: Dict[str, Callable] = {}

def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        # Preserve the original entry so tests mucking with ``REGISTRY`` can be
        # reset to a known-good state.
        _REGISTRY_BASE.setdefault(name, fn)
        return fn
    return deco


def reset_registry() -> None:
    """Restore :data:`REGISTRY` to the original registered terms."""
    REGISTRY.clear()
    REGISTRY.update(_REGISTRY_BASE)

def enabled_terms(cfg: Dict[str, Any], phase: str) -> List[str]:
    """
    基于 cfg 的“权重门控”选择本轮启用的 term 列表。
    规则：只有当该项的 k>0 时才启用对应 term；否则不启用。
    - 在单元测试或快速实验中，只需在 cfg 里把不想参与的项的 k 设为 0（或不提供该键）。
    - 例如：只测 anchor → {"anchor.k.spring": 10.0}（其它 k 默认=0，不会启用）
    """
    terms: List[str] = []
    if phase == "pre_anchor":
        # label↔label（仅在 repulse/inside 有正权重时启用）
        if float(cfg.get("ll.k.repulse", 0)) > 0 or float(cfg.get("ll.k.inside", 0)) > 0:
            geom = cfg.get("ll.geom", "rect")
            terms.append("ll.rect" if geom == "rect" else "ll.disk")

        # 边界（画幅四墙）
        if float(cfg.get("boundary.k.wall", 0)) > 0:
            terms.append("boundary.wall")

        # 点-Label、线-Label
        if float(cfg.get("pl.k.repulse", 0)) > 0 or float(cfg.get("pl.k.inside", 0)) > 0:
            terms.append("pl.rect")
        if float(cfg.get("ln.k.repulse", 0)) > 0 or float(cfg.get("ln.k.inside", 0)) > 0:
            terms.append("ln.rect")

        # 面相关
        if float(cfg.get("area.k.embed", 0)) > 0:
            terms.append("area.embed")
        if float(cfg.get("area.k.cross", 0)) > 0:
            terms.append("area.cross")
        if float(cfg.get("area.k.softout", 0)) > 0:
            terms.append("area.softout")

        # focus 场（吸引/定向）
        if float(cfg.get("focus.k.attract", 0)) > 0:
            terms.append("focus.attract")
        return terms

    if phase == "anchor":
        # 锚点弹簧
        if float(cfg.get("anchor.k.spring", 0)) > 0:
            terms.append("anchor.spring")
        return terms

    return terms

# 显式导入各 term 模块以触发注册（很重要，否则 @register 不会执行）
from . import term_focus
from . import term_boundary           # "boundary.wall"
from . import term_anchor             # "anchor.spring"
from . import term_label_label_rect   # "ll.rect"
from . import term_label_label_disk   # "ll.disk"
from . import term_point_label        # "pl.rect"
from . import term_line_label         # "ln.rect"
from . import term_area_embed         # "area.embed"
from . import term_area_cross         # "area.cross"
from . import term_area_softout       # "area.softout"
