"""Spring force term ('anchor.spring'): pulls P toward resolved anchor A. Physics lives here; geometry resolution lives in resolver."""
from __future__ import annotations

import numpy as np

from cartoweave.compute.geom_anchor_resolver import anchor_position
from cartoweave.utils.compute_common import ensure_vec2, get_eps
from cartoweave.utils.kernels import EPS_NORM
from cartoweave.utils.logging import logger

from . import register


def _val(lab, key, default=None):
    """通用字段读取：兼容 dict 和 LabelState。
       - 支持 'kind' / 'mode' / 其它 meta 字段（mode 会从 meta 提升）
    """
    if isinstance(lab, dict):
        if key == "mode":
            return lab.get("mode") or (lab.get("meta") or {}).get("mode", default)
        return lab.get(key, default)
    # LabelState
    if key == "mode":
        m = getattr(lab, "meta", None)
        return (m or {}).get("mode", default)
    return getattr(lab, key, default)


@register("anchor.spring")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    L = P.shape[0] if P is not None else 0
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "anchors"}

    import logging
    log = logging.getLogger(__name__)

    labels_all = scene.get("labels", []) or []
    pts   = scene.get("points", []) or []
    lines = scene.get("lines",  []) or []     # 你的 scene 里是 polyline 列表
    areas = scene.get("areas",  []) or []     # 你的 scene 里是坐标序列列表

    # --- 与 P 完全对齐：优先取 _active_ids_solver，其次 _active_ids；否则回退为身份映射 ---
    N = P.shape[0]
    all_active = scene.get("_active_ids_solver") or scene.get("_active_ids")
    if not all_active or len(all_active) != N:
        log.warning(
            "anchors: active ids len=%s != P rows=%d; fallback to identity [0..N-1]",
            None if all_active is None else len(all_active), N
        )
        all_active = list(range(N))

    # 与 P 同序取 labels；越界时提供空壳，避免字段访问炸
    def _get_label(i):
        if 0 <= i < len(labels_all):
            return labels_all[i]
        class _Dummy:
            kind = None
            WH = (0.0, 0.0)
            anchor = None
            def __getattr__(self, _): return None
        return _Dummy()

    labels = [_get_label(i) for i in all_active]

    # --- 工具：把 anchor 解析成 (target, index, xy) 且结构稳健 ---
    def _parse_anchor(lab):
        a = getattr(lab, "anchor", None)
        if a is None:
            return None
        # 你的 Label.anchor 看起来是 Anchor(target='point'|... , index=int, mode=..., xy=None|tuple)
        tgt  = getattr(a, "target", None) or (a.get("target") if isinstance(a, dict) else None)
        idx  = getattr(a, "index",  None) or (a.get("index")  if isinstance(a, dict) else None)
        xy   = getattr(a, "xy",     None) or (a.get("xy")     if isinstance(a, dict) else None)
        return {"target": tgt, "index": idx, "xy": xy}

    # 输出向量与能量
    F = np.zeros_like(P)
    E = 0.0

    # ---- 你原有的参数读取保持不变 ----
    k = float(cfg.get("anchor.k", 0.0))
    if k <= 0.0:
        return 0.0, ensure_vec2(F, L), {"term": "anchors", "disabled": True}

    # 主循环：逐行（与 P 对齐），根据 anchor.target 去不同几何源取参考点
    for i, lab in enumerate(labels):
        a = _parse_anchor(lab)
        if not a:
            continue

        tgt = a["target"]
        idx = a["index"]
        # 允许 xy 覆盖，若给定则直接用
        if a["xy"] is not None:
            x_ref, y_ref = float(a["xy"][0]), float(a["xy"][1])
        elif tgt == "point" and idx is not None and 0 <= idx < len(pts):
            x_ref, y_ref = float(pts[idx][0]), float(pts[idx][1])
        elif tgt == "line" and idx is not None and 0 <= idx < len(lines):
            # 线锚：常见做法取最近点/投影点；此处如果你已有 helper，就调用，否则先取折线某个代表点
            poly = np.asarray(lines[idx], float).reshape(-1, 2)
            if poly.shape[0] == 0:
                continue
            # 先用折线“最近顶点”作为参考（TODO：可替换为投影到 polyline 的最近点）
            j = int(np.argmin(np.sum((poly - P[i])**2, axis=1)))
            x_ref, y_ref = float(poly[j,0]), float(poly[j,1])
        elif tgt == "area" and idx is not None and 0 <= idx < len(areas):
            # 面锚：可取最近边/最近点；暂用多边形质心作为参考
            poly = np.asarray(areas[idx], float).reshape(-1, 2)
            if poly.shape[0] < 3:
                continue
            x_ref, y_ref = float(np.mean(poly[:,0])), float(np.mean(poly[:,1]))
        else:
            # 未识别或越界
            continue

        # 弹簧到锚（保持你原有的能量/梯度形式；这里给一个典型的 Hooke 形式）
        dx, dy = float(P[i,0]) - x_ref, float(P[i,1]) - y_ref
        E += 0.5 * k * (dx*dx + dy*dy)
        F[i,0] += -k * dx
        F[i,1] += -k * dy

    return float(E), ensure_vec2(F, L), {"term": "anchor.spring"}
