"""Polygon utilities and generators."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..sampling.helpers import inset_rect, project_to_rect_inset, frame_metrics


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float) -> bool:
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float) -> bool:
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    if abs(o1) < eps and _on_segment(a, b, c, eps):
        return True
    if abs(o2) < eps and _on_segment(a, b, d, eps):
        return True
    if abs(o3) < eps and _on_segment(c, d, a, eps):
        return True
    if abs(o4) < eps and _on_segment(c, d, b, eps):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def segments_self_intersect(poly: np.ndarray, eps: float = 1e-9) -> bool:
    """Return ``True`` if any two segments of ``poly`` intersect."""
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("poly must be of shape (N,2)")
    pts = np.asarray(poly, float)
    n = pts.shape[0]
    if n < 4:
        return False
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if _segments_intersect(segs[i][0], segs[i][1], segs[j][0], segs[j][1], eps):
                return True
    return False


def polygon_self_intersect(poly: np.ndarray, eps: float = 1e-9) -> bool:
    """Robust self-intersection test for polygons.

    ``poly`` may be open or closed; the closing edge is always assumed during
    the intersection test.
    """
    pts = np.asarray(poly, float)
    if pts.shape[0] < 4:
        return False
    return segments_self_intersect(pts, eps=eps)


def generate_polygon_by_area(
    rng: np.random.Generator,
    frame_size: tuple[int, int],
    S_target: float,
    inset_margin: float,
    min_edge_spacing: float,
    n_vertices_range: tuple[int, int] = (6, 14),
    max_tries: int = 32,   # 用作“初始化重试”上限
    eps: float = 1e-9,
) -> np.ndarray:
    import math
    import numpy as np

    W, H = frame_size
    x0, y0 = inset_margin, inset_margin
    x1, y1 = W - inset_margin, H - inset_margin
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)

    # —— 中心采样加缓冲，至少离边 ~3×spacing 或内框 6% ——
    pad_x = max(3.0 * min_edge_spacing, 0.06 * (x1 - x0))
    pad_y = max(3.0 * min_edge_spacing, 0.06 * (y1 - y0))

    Cx = float(rng.uniform(x0 + pad_x, x1 - pad_x)) if (x1 - x0) > 2 * pad_x else 0.5 * (x0 + x1)
    Cy = float(rng.uniform(y0 + pad_y, y1 - pad_y)) if (y1 - y0) > 2 * pad_y else 0.5 * (y0 + y1)

    # 本地半径（到四边的空间），并设硬下限，避免 1~2px
    rx = max(Cx - x0, x1 - Cx)
    ry = max(Cy - y0, y1 - Cy)
    rmin = max(2.5 * min_edge_spacing, 8.0)
    rx = max(rx, rmin);
    ry = max(ry, rmin)

    kmin, kmax = int(n_vertices_range[0]), int(n_vertices_range[1])
    kmin = max(3, kmin)
    if kmax < kmin:
        kmax = kmin

    def poly_area(p: np.ndarray) -> float:
        x, y = p[:, 0], p[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # ✅ 只收缩不放大，且用本地 rx, ry
    def fit_into_rect_about(p: np.ndarray, Cx: float, Cy: float) -> np.ndarray:
        dx = float(np.max(np.abs(p[:, 0] - Cx)))
        dy = float(np.max(np.abs(p[:, 1] - Cy)))
        if dx < 1e-12 and dy < 1e-12:
            return p
        sx = rx / max(dx, 1e-12)
        sy = ry / max(dy, 1e-12)
        s = min(1.0, 0.99 * min(sx, sy))  # 只收缩，不放大
        q = (p - (Cx, Cy)) * s + (Cx, Cy)
        # 收缩后半径不得小于 rmin（防止“缩没”）
        dx2 = float(np.max(np.abs(q[:, 0] - Cx)));
        dy2 = float(np.max(np.abs(q[:, 1] - Cy)))
        if max(dx2, dy2) < rmin:
            s_back = rmin / max(max(dx2, dy2), 1e-9)
            q = (q - (Cx, Cy)) * s_back + (Cx, Cy)
        return q

    def check_spacing(p: np.ndarray, dmin: float) -> bool:
        q = np.vstack([p, p[0]])
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        if np.any(seg < dmin - eps):
            return False
        k = len(p)
        for i in range(k):
            for j in range(i + 2, k):
                if i == 0 and j == k - 1:
                    continue
                if np.linalg.norm(p[i] - p[j]) < dmin - eps:
                    return False
        return True

    def random_convex_polygon_valtr(k: int) -> np.ndarray:
        xs = np.sort(rng.uniform(-1.0, 1.0, size=k))
        ys = np.sort(rng.uniform(-1.0, 1.0, size=k))
        dxs = np.diff(np.concatenate([[xs[0]], xs, [xs[-1]]]))
        dys = np.diff(np.concatenate([[ys[0]], ys, [ys[-1]]]))
        rng.shuffle(dxs); rng.shuffle(dys)
        vecs = np.stack([dxs - dxs.mean(), dys - dys.mean()], axis=1)
        order = np.argsort(np.arctan2(vecs[:, 1], vecs[:, 0]))
        pts = np.cumsum(vecs[order], axis=0)
        pts -= pts.mean(axis=0)
        # 轻微各向缩放，避免过圆
        sx0, sy0 = rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0)
        pts[:, 0] *= sx0; pts[:, 1] *= sy0
        # ✅ 初始尺度基于本地半径 rx, ry（避免一上来就越界）
        mx = np.max(np.abs(pts[:, 0])) + 1e-9
        my = np.max(np.abs(pts[:, 1])) + 1e-9
        s0 = 0.8 * min(rx / mx, ry / my)
        pts[:, 0] = pts[:, 0] * s0 + Cx
        pts[:, 1] = pts[:, 1] * s0 + Cy
        return pts

    def add_concavities_once(p: np.ndarray, frac: float = 0.25) -> np.ndarray:
        k = len(p)
        if k < 5 or frac <= 0:
            return p
        m = max(1, int(round(k * frac)))
        idxs = rng.choice(k, size=m, replace=False)
        c = p.mean(axis=0)
        q = p.copy()
        for i in idxs:
            v = q[i] - c; nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue
            im1, ip1 = (i - 1) % k, (i + 1) % k
            tan = q[ip1] - q[im1]; nt = np.linalg.norm(tan)
            if nt > 1e-9: tan /= nt
            inward = rng.uniform(0.08, 0.25)
            t_jit  = rng.uniform(-0.05, 0.05)
            cand = q[i] - inward * v + t_jit * min_edge_spacing * tan
            tmp = q.copy(); tmp[i] = cand
            if (x0 <= cand[0] <= x1 and y0 <= cand[1] <= y1 and
                not polygon_self_intersect(tmp, eps=eps) and
                check_spacing(tmp, min_edge_spacing)):
                q = tmp
        return q

    # === 1) 初始化（最多重试几次） ===
    for _init in range(max_tries):
        k = int(rng.integers(kmin, kmax + 1))
        pts = random_convex_polygon_valtr(k)

        # 面积匹配（绕 C）—— ✅ 放大时必须“夹到允许上界”
        A = abs(poly_area(pts))
        if A > 1e-12:
            dx = np.max(np.abs(pts[:, 0] - Cx)); dy = np.max(np.abs(pts[:, 1] - Cy))
            sA = math.sqrt(max(abs(S_target), 1e-12) / A)
            allow_up = 0.99 * min(rx / max(dx, 1e-12), ry / max(dy, 1e-12))
            if sA > 1.0:
                sA = min(sA, max(0.1, allow_up))   # 不能超过允许上界，避免越界后被极小比例“缩没”
            pts = (pts - (Cx, Cy)) * sA + (Cx, Cy)

        pts = fit_into_rect_about(pts, Cx, Cy)

        if check_spacing(pts, min_edge_spacing) and not polygon_self_intersect(pts, eps=eps):
            break
    else:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)

    # 可选：一次凹陷
    pts = add_concavities_once(pts, frac=rng.uniform(0.15, 0.30))
    # 再微调面积（同样夹上界）+ 进框
    A = abs(poly_area(pts))
    if A > 1e-12:
        dx = np.max(np.abs(pts[:, 0] - Cx)); dy = np.max(np.abs(pts[:, 1] - Cy))
        sA = math.sqrt(max(abs(S_target), 1e-12) / A)
        allow_up = 0.99 * min(rx / max(dx, 1e-12), ry / max(dy, 1e-12))
        if sA > 1.0:
            sA = min(sA, max(0.1, allow_up))
        pts = (pts - (Cx, Cy)) * sA + (Cx, Cy)
    pts = fit_into_rect_about(pts, Cx, Cy)

    # === 2) 细化循环（与你的逻辑相同，但把入框改成“绕 C 且只收缩”） ===
    refine_iters = 400
    step = 1.2 * min_edge_spacing
    step_min = 0.05 * min_edge_spacing
    accept_boost, reject_decay = 1.05, 0.7
    patience, no_improve = 60, 0

    def area_err(p):
        return abs(abs(poly_area(p)) - abs(S_target))

    best = pts.copy()
    best_err = area_err(best)

    for it in range(refine_iters):
        i = int(rng.integers(0, len(pts)))
        im1, ip1 = (i - 1) % len(pts), (i + 1) % len(pts)

        tan = pts[ip1] - pts[im1]
        nt = np.linalg.norm(tan)
        if nt < 1e-12:
            continue
        tan /= nt
        nor = np.array([-tan[1], tan[0]])
        sign_area = 1.0 if poly_area(pts) < S_target else -1.0
        delta = (0.6 * sign_area * nor + 0.4 * rng.normal(0.0, 1.0) * tan) * step

        cand = pts.copy()
        cand[i] = cand[i] + delta
        cand = fit_into_rect_about(cand, Cx, Cy)  # ✅ 只收缩入框

        if polygon_self_intersect(cand, eps=eps):
            step = max(step * reject_decay, step_min); no_improve += 1; continue
        if not check_spacing(cand, min_edge_spacing):
            step = max(step * reject_decay, step_min); no_improve += 1; continue

        e_old = area_err(pts)
        e_new = area_err(cand)
        accept = e_new <= e_old or rng.uniform() < math.exp(-(e_new - e_old) / max(e_old, 1e-9))
        if accept:
            pts = cand
            step *= accept_boost
            if e_new < best_err:
                best, best_err = pts.copy(), e_new
            no_improve = 0
        else:
            step = max(step * reject_decay, step_min); no_improve += 1

        if (it + 1) % 25 == 0:
            A = abs(poly_area(pts))
            if A > 1e-12:
                dx = np.max(np.abs(pts[:, 0] - Cx)); dy = np.max(np.abs(pts[:, 1] - Cy))
                sA = math.sqrt(max(abs(S_target), 1e-12) / A)
                allow_up = 0.99 * min(rx / max(dx, 1e-12), ry / max(dy, 1e-12))
                if sA > 1.0:
                    sA = min(sA, max(0.1, allow_up))
                pts = (pts - (Cx, Cy)) * sA + (Cx, Cy)
            pts = fit_into_rect_about(pts, Cx, Cy)
            if polygon_self_intersect(pts, eps=eps) or not check_spacing(pts, min_edge_spacing):
                pts = best.copy()

        if no_improve > patience:
            step = min(2.0 * min_edge_spacing, step * 1.3)
            no_improve = 0

        if best_err / max(abs(S_target), 1e-9) < 0.02:
            return best.astype(float)

    return best.astype(float)
