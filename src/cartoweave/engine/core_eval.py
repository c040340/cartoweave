# src/cartoweave/engine/core_eval.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Callable
import os
import numpy as np
from cartoweave.utils.checks import merge_sources, check_force_grad_consistency
from cartoweave.utils.logging import logger

from .forces import REGISTRY, enabled_terms, reset_registry
from ..utils.shape import as_nx2


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
    reset_registry()
    P = np.asarray(P, dtype=float)
    assert P.ndim == 2 and P.shape[1] == 2, f"P must be (N,2), got {P.shape}"
    g = np.zeros_like(P)
    comps: Dict[str, np.ndarray] = {}
    sources_merged: Dict[str, Any] = {}
    E_total = 0.0

    debug_forces = (
        os.environ.get("CFG_DEBUG_FORCES", "") not in {"", "0", "false", "False"}
        or bool(
            cfg.get("solver", {})
            .get("tuning", {})
            .get("debug", {})
            .get("force_stats")
        )
    )
    term_norms: Dict[str, float] | None = None
    ext_norm = 0.0
    if debug_forces and not cfg.get("_force_stats_logged"):
        term_norms = {}

    clamp_max = (
        cfg.get("solver", {})
        .get("tuning", {})
        .get("clamp", {})
        .get("optimize_force_max")
    )
    if clamp_max is not None:
        clamp_max = float(clamp_max)

    labels_all = scene.get("labels", [])
    active_ids = scene.get("_active_ids", list(range(len(labels_all))))
    modes = [labels_all[i].get("mode") for i in active_ids if i < len(labels_all)]
    circle_count = sum(m == "circle" for m in modes)
    if scene.pop("_log_label_stats", False):
        logger.debug(
            "enabled_terms: active=%d circles=%d", len(active_ids), circle_count
        )

    pre_terms = list(enabled_terms(cfg, phase="pre_anchor"))
    if "ll.rect" in pre_terms and circle_count == len(active_ids) and len(active_ids) > 0:
        pre_terms.remove("ll.rect")
    anchor_terms = list(enabled_terms(cfg, phase="anchor"))

    # (1) 先算所有“非 anchor”项，累计 E 与 g，同时累计外力合力 Fsum_ext
    Fsum_ext = np.zeros_like(P)
    for name in pre_terms:
        term = REGISTRY[name]
        E_add, F_add, source = term(scene, P, cfg, phase="pre_anchor")
        E_total += float(E_add)
        if F_add is not None:
            if F_add.shape != P.shape:
                step_name = scene.get("_current_step_name")
                stage_name = scene.get("_current_stage_name")
                raise ValueError(
                    f"[TERM SHAPE MISMATCH] term={name} F_add={F_add.shape} P={P.shape} "
                    f"step={step_name} stage={stage_name}"
                )
            if clamp_max is not None:
                np.clip(F_add, -clamp_max, clamp_max, out=F_add)
            g -= F_add  # F = -∇E  →  ∇E 累加为 -F
            comps[name] = F_add
            Fsum_ext += F_add
            if term_norms is not None:
                term_norms[name] = float(np.linalg.norm(F_add))
        if source:
            merge_sources(sources_merged, source)

    # (2) 把“外力合力方向”传给 anchor（仅在 r≈0 时会用到）
    # 与旧工程一致：通过 data/scene["_ext_dir"] 提供兜底方向。:contentReference[oaicite:1]{index=1}
    scene["_ext_dir"] = Fsum_ext.copy()
    if term_norms is not None:
        ext_norm = float(np.linalg.norm(Fsum_ext))

    # (3) 再算 anchor 项（最后阶段）
    for name in anchor_terms:
        term = REGISTRY[name]
        E_add, F_add, source = term(scene, P, cfg, phase="anchor")
        E_total += float(E_add)
        if F_add is not None:
            if F_add.shape != P.shape:
                step_name = scene.get("_current_step_name")
                stage_name = scene.get("_current_stage_name")
                raise ValueError(
                    f"[TERM SHAPE MISMATCH] term={name} F_add={F_add.shape} P={P.shape} "
                    f"step={step_name} stage={stage_name}"
                )
            if clamp_max is not None:
                np.clip(F_add, -clamp_max, clamp_max, out=F_add)
            g -= F_add
            comps[name] = F_add
            if term_norms is not None:
                term_norms[name] = float(np.linalg.norm(F_add))
        if source:
            merge_sources(sources_merged, source)

    # 清理临时键（与旧工程一致）:contentReference[oaicite:2]{index=2}
    scene.pop("_ext_dir", None)

    if term_norms is not None:
        cfg["_force_stats_logged"] = True
        top_terms = sorted(term_norms.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_str = ", ".join(f"{k}={v:.3g}" for k, v in top_terms)
        logger.info(f"[forces] ext_norm={ext_norm}, top_terms: {top_str}")
        if ext_norm < 1e-12:
            logger.info("[forces] resultant is ~0 (check threshold/weights/presets)")

    # debug 一致性检查（可选）
    if bool(cfg.get("debug.check", False)):
        Fsum = np.zeros_like(P)
        for V in comps.values():
            if isinstance(V, np.ndarray):
                Fsum += V
        check_force_grad_consistency(Fsum, g)
        # 这里不做日志依赖，调用方可按需要打印结果

    # 记录钩子：保留每帧 P/E/分力分解/sources（与旧工程的 record 协议一致）:contentReference[oaicite:3]{index=3}
    if record is not None:
        comps_copy = {k: np.asarray(v, float).copy() for k, v in comps.items()}
        meta = {"core": "energy_and_grad_fullP"}
        if sources_merged:
            meta.update(sources_merged)
        record(P.copy(), float(E_total), comps_copy, meta)

    return float(E_total), g, {"sources": sources_merged}

def scalar_potential_field(scene: dict,
                           P: np.ndarray,
                           cfg: dict,
                           *,
                           label_index: int = 0,
                           resolution: int | None = None,
                           mode: str = "grad_norm"  # "energy" | "grad_norm"
                           ) -> np.ndarray:
    """
    Build a scalar field over the frame by probing the given label's position
    across a grid and evaluating either total energy ("energy") or the gradient
    magnitude at that label ("grad_norm"). Returned array has shape (ny, nx).

    Key fix (2025-09-05):
      - Per-label arrays (e.g., WH) must be validated/broadcast against L (#labels),
        NOT against M (#grid samples). This function enforces L for WH, etc.

    Args
    ----
    scene : dict
        Scene dictionary containing frame_size, geometry sources, per-label params, etc.
        Must include (or allow default) "WH" for label rectangles (shape (L,2) or (2,)).
    P : (L,2) float ndarray
        Current label centers.
    cfg : dict
        Global/solver config. May contain viz.field.resolution or similar.
    label_index : int
        Which label to probe over the grid.
    resolution : int | None
        If None, choose a reasonable ny based on frame height, keeping aspect ratio.
        If int, used as ny directly (nx is scaled by aspect).
    mode : {"energy","grad_norm"}
        - "grad_norm": compute ||∂E/∂P_i|| for the probed label (default,直观显示“陡峭/平坦”)
        - "energy"   : compute total E at that probe position

    Returns
    -------
    field : (ny, nx) float ndarray
        Scalar field suitable for heatmap/contour rendering. NaN-free.
    """
    # ------ sanitize inputs ------
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"P must be (L,2); got shape={P.shape}")

    L = int(P.shape[0])                         # number of labels
    if L == 0:
        return np.zeros((1, 1), dtype=float)

    # clamp label index
    li = int(max(0, min(label_index, L - 1)))

    # ------ frame size & grid resolution ------
    # frame_size could be (W,H) or stored under scene["frame_size"] etc.
    fs = scene.get("frame_size", None)
    if isinstance(fs, (list, tuple)) and len(fs) == 2:
        W, H = int(fs[0]), int(fs[1])
    else:
        # try cfg fallback
        fs2 = cfg.get("frame_size", (1920, 1080))
        W, H = int(fs2[0]), int(fs2[1])

    W = max(int(W), 4)
    H = max(int(H), 4)

    # decide grid resolution (ny rows, nx cols) keeping aspect
    if resolution is None:
        # default ny ~ min( max(80, H/12), 220 ) as a reasonable balance
        ny = int(max(80, min(220, round(H / 12))))
    else:
        ny = int(max(8, resolution))

    nx = max(8, int(round(ny * (W / float(H)))))  # keep aspect ratio

    # ------ per-label arrays: enforce L (NOT M) ------
    # WH (label rectangle sizes), allow (2,) or (1,2) to broadcast to (L,2)
    WH_raw = scene.get("WH", (10.0, 5.0))
    WH = as_nx2(WH_raw, L, "WH")  # <<<<<< KEY FIX: validate against L

    # If you have other per-label tensors that this function needs, validate them here with L.
    # For example (uncomment/adjust if your project uses them in the eval path):
    # labels_init_raw = scene.get("labels_init", P)
    # labels_init = as_nx2(labels_init_raw, L, "labels_init")

    # ------ build probing grid in pixel coords ------
    ys = np.linspace(0.5, H - 0.5, ny, dtype=float)  # center of pixels
    xs = np.linspace(0.5, W - 0.5, nx, dtype=float)

    field = np.zeros((ny, nx), dtype=float)

    # ------ evaluation scratch ------
    # We keep a shallow copy of scene to patch label-wise arrays if needed.
    sc = dict(scene)
    sc["WH"] = WH  # ensure L-consistent in the evaluator path

    # Pre-allocate a working copy of P to reduce reallocations
    P_work = np.array(P, dtype=float, copy=True)

    # ------ evaluation scratch ------
    sc = dict(scene)
    sc["WH"] = WH  # 已按 L 校验

    # [patch-yx-2025-09-05] make active_ids consistent with P during field probing
    L = int(P.shape[0])
    all_active = list(range(L))
    # 覆盖任何旧的/步态相关的 active 集，确保长度与 P 匹配
    sc["_active_ids_solver"] = all_active
    # 有些实现还会读取这个键（如果存在）
    sc["_active_ids"] = all_active

    # ------ choose evaluator ------
    use_grad_norm = (mode == "grad_norm")
    # Expect energy_and_grad_fullP(scene, P, cfg) -> (E, G, *rest)
    # where G has shape (L,2). Adjust name if your evaluator differs.

    for j, y in enumerate(ys):
        # set row-wise once to avoid attribute lookups in inner loop
        for i, x in enumerate(xs):
            # move only the probed label to (x,y)
            P_work[li, 0] = x
            P_work[li, 1] = y

            try:
                out = energy_and_grad_fullP(sc, P_work, cfg)
            except TypeError:
                # Some implementations return fixed arity; handle safely
                E, G = out[0], out[1] if isinstance(out, (list, tuple)) else (float(out), None)
            else:
                # Normal case
                if isinstance(out, (list, tuple)):
                    if len(out) >= 2:
                        E, G = out[0], out[1]
                    else:
                        E, G = out[0], None
                else:
                    E, G = float(out), None

            if use_grad_norm and (G is not None):
                gxy = np.asarray(G, dtype=float)
                if gxy.shape == (L, 2) and np.all(np.isfinite(gxy[li])):
                    val = float(np.hypot(gxy[li, 0], gxy[li, 1]))
                else:
                    # fallback if gradient missing/bad
                    val = float(E) if np.isfinite(E) else 0.0
            else:
                # energy mode or missing gradient
                val = float(E) if np.isfinite(E) else 0.0

            field[j, i] = val

    # ------ sanitize numeric issues ------
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional dynamic range conditioning (keeps colormap useful)
    vmin = field.min()
    vmax = field.max()
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        # mild compression to avoid single spikes dominating
        # (you can tweak/remove as you prefer)
        q_hi = np.quantile(field, 0.995)
        field = np.minimum(field, q_hi)

    return field

