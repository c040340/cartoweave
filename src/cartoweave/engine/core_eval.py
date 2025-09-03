# src/cartoweave/engine/core_eval.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Callable
import numpy as np
from cartoweave.utils.checks import merge_sources, check_force_grad_consistency

from .forces import REGISTRY, enabled_terms


def energy_and_grad_fullP(
    scene: Dict[str, Any],
    P: np.ndarray,
    cfg: Dict[str, Any],
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None] | None = None,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    计算总能量与梯度（对完整 P=(N,2)）。
    - 返回: (E_total, G (N,2), info_dict)；其中 info_dict 至少包含 {"sources": dict}
    - 评估顺序：先非 anchor → 汇总外力方向 → anchor。anchor 会读取 scene["_ext_dir"] 做 r≈0 兜底。
      （顺序与旧工程相同）
    """
    P = np.asarray(P, dtype=float)
    g = np.zeros_like(P)
    comps: Dict[str, np.ndarray] = {}
    sources_merged: Dict[str, Any] = {}
    E_total = 0.0

    # 选择启用的力项（名称与注册表一致）
    names = tuple(enabled_terms(cfg, phase="pre_anchor")) + tuple(enabled_terms(cfg, phase="anchor"))

    # (1) 先算所有“非 anchor”项，累计 E 与 g，同时累计外力合力 Fsum_ext
    Fsum_ext = np.zeros_like(P)
    for name in enabled_terms(cfg, phase="pre_anchor"):
        term = REGISTRY[name]
        E_add, F_add, source = term(scene, P, cfg, phase="pre_anchor")
        E_total += float(E_add)
        if F_add is not None:
            g -= F_add  # F = -∇E  →  ∇E 累加为 -F
            comps[name] = F_add
            Fsum_ext += F_add
        if source:
            merge_sources(sources_merged, source)

    # (2) 把“外力合力方向”传给 anchor（仅在 r≈0 时会用到）
    # 与旧工程一致：通过 data/scene["_ext_dir"] 提供兜底方向。:contentReference[oaicite:1]{index=1}
    scene["_ext_dir"] = Fsum_ext.copy()

    # (3) 再算 anchor 项（最后阶段）
    for name in enabled_terms(cfg, phase="anchor"):
        term = REGISTRY[name]
        E_add, F_add, source = term(scene, P, cfg, phase="anchor")
        E_total += float(E_add)
        if F_add is not None:
            g -= F_add
            comps[name] = F_add
        if source:
            merge_sources(sources_merged, source)

    # 清理临时键（与旧工程一致）:contentReference[oaicite:2]{index=2}
    scene.pop("_ext_dir", None)

    # debug 一致性检查（可选）
    if bool(cfg.get("debug.check", False)):
        Fsum = np.zeros_like(P)
        for V in comps.values():
            if isinstance(V, np.ndarray):
                Fsum += V
        err = check_force_grad_consistency(Fsum, g)
        # 这里不做日志依赖，调用方可按需要打印 err

    # 记录钩子：保留每帧 P/E/分力分解/sources（与旧工程的 record 协议一致）:contentReference[oaicite:3]{index=3}
    if record is not None:
        comps_copy = {k: v.copy() for k, v in comps.items()}
        record(P.copy(), float(E_total), comps_copy, sources_merged)

    return float(E_total), g, {"sources": sources_merged}


def scalar_potential_field(
    scene: Dict[str, Any],
    P: np.ndarray,
    cfg: Dict[str, Any],
    *,
    label_index: int = 0,
    resolution: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """Evaluate a scalar potential field for one label on a grid.

    This helper computes the total energy of the scene while probing the
    position of a single label over a regular grid.  The other labels remain
    fixed.  The result is an array ``(H, W)`` where ``H``/``W`` correspond to the
    number of samples in the vertical and horizontal directions respectively.

    Parameters
    ----------
    scene:
        Scene description as used by :func:`energy_and_grad_fullP`.
    P:
        Current label positions ``(N, 2)``.
    cfg:
        Configuration dictionary.
    label_index:
        Index of the label to probe.  Defaults to ``0``.
    resolution:
        Either an integer specifying the number of samples for both axes or a
        tuple ``(ny, nx)`` with individual resolutions.  If ``None`` (the
        default), the value of ``cfg["viz.field.resolution"]`` is used when
        present, otherwise ``100``.
    """

    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("P must be of shape (N,2)")

    if resolution is None:
        resolution = cfg.get("viz.field.resolution", 100)

    if isinstance(resolution, int):
        ny = nx = int(resolution)
    else:
        ny, nx = map(int, resolution)

    W, H = scene.get("frame_size", (1.0, 1.0))
    xs = np.linspace(0.0, float(W), nx)
    ys = np.linspace(0.0, float(H), ny)

    base = P.copy()
    E0, _, _ = energy_and_grad_fullP(scene, base, cfg)

    field = np.zeros((ny, nx), dtype=float)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            probe = base.copy()
            probe[label_index] = (x, y)
            E, _, _ = energy_and_grad_fullP(scene, probe, cfg)
            field[iy, ix] = E - E0

    return field
