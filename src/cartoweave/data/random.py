# src/cartoweave/data/random.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import os, time, math, string, random
import numpy as np
from ..labels import anchor_xy, init_position

Array = np.ndarray

# ---------- 非物理键，允许默认值 ----------
_DEF_CACHE_NAME = "scene_cache.npz"
_DEF_CANVAS = (1080.0, 1920.0)


def _frame_metrics(frame_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return (W, H, D, A) where D is diagonal, A is W*H (float)."""
    W, H = frame_size
    D = (W ** 2 + H ** 2) ** 0.5
    A = float(W) * float(H)
    return float(W), float(H), float(D), float(A)


def _sample_split_normal_trunc(mu: float, L: float, U: float, k: int, rng: np.random.Generator) -> float:
    # ensure mu inside (L,U)
    tiny = 1e-6 * (U - L)
    mu = min(max(mu, L + tiny), U - tiny)
    sigL = max((mu - L) / float(k), 1e-12 * (U - L))
    sigR = max((U - mu) / float(k), 1e-12 * (U - L))
    pL = sigL / (sigL + sigR)
    for _ in range(128):
        if rng.random() < pL:
            x = mu - abs(rng.normal(0.0, sigL))
        else:
            x = mu + abs(rng.normal(0.0, sigR))
        if L <= x <= U:
            return float(x)
    return float(np.clip(mu, L, U))


@dataclass
class RouteGenCfg:
    mean_length_scale: float = 0.55
    k_sigma_bound: int = 4
    min_vertex_spacing_scale: float = 0.01
    min_edge_margin_scale: float = 0.02
    lower_bound_scale: float = 0.10
    upper_bound_scale: float = 1.20


@dataclass
class AreaGenCfg:
    mean_area_scale: float = 0.05
    k_sigma_bound: int = 5
    min_vertex_spacing_scale: float = 0.01
    min_edge_margin_scale: float = 0.02
    lower_bound_scale: float = 0.01
    upper_bound_scale: float = 0.50


def _sample_route_length(frame_size, cfg: RouteGenCfg, rng) -> float:
    W, H, D, A = _frame_metrics(frame_size)
    mu = cfg.mean_length_scale * D
    L = cfg.lower_bound_scale * D
    U = cfg.upper_bound_scale * D
    return _sample_split_normal_trunc(mu, L, U, cfg.k_sigma_bound, rng)


def _sample_area_size(frame_size, cfg: AreaGenCfg, rng) -> float:
    W, H, D, A = _frame_metrics(frame_size)
    mu = cfg.mean_area_scale * A
    L = cfg.lower_bound_scale * A
    U = cfg.upper_bound_scale * A
    return _sample_split_normal_trunc(mu, L, U, cfg.k_sigma_bound, rng)

# =============== 随机与缓存 ===============
def _apply_seed(seed: Optional[int]) -> int:
    """同步 numpy / random 的随机种子；返回实际种子。"""
    s = int(time.time()) if seed is None else int(seed)
    np.random.seed(s)
    random.seed(s)
    return s

def _default_cache_path(filename: str = _DEF_CACHE_NAME) -> str:
    """默认缓存地址：项目根的 .cache 目录（可替换为更高级的 platformdirs）。"""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    cache_dir = os.path.join(root, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)

def save_scene(data: Dict[str, Any], path: str) -> None:
    """Save scene dict using ``np.savez`` (allows storing arrays directly)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, data=np.array(data, dtype=object))


def load_scene(path: str) -> Dict[str, Any]:
    """Load scene dict produced by :func:`save_scene`."""
    obj = np.load(path, allow_pickle=True)["data"].item()
    return obj

# =============== 基础几何采样 ===============
def _canvas_focus_box(canvas_size: Tuple[float, float], focus_ratio: float):
    W, H = canvas_size
    fw, fh = W * focus_ratio, H * focus_ratio
    x0 = 0.5 * (W - fw); y0 = 0.5 * (H - fh)
    return x0, y0, x0 + fw, y0 + fh

def _poisson_disc(canvas_size, r_min=40, k=20, rng=np.random) -> List[Tuple[float,float]]:
    W, H = canvas_size
    cell = r_min / math.sqrt(2.0)
    gw, gh = int(math.ceil(W/cell)), int(math.ceil(H/cell))
    grid = [[-1]*gw for _ in range(gh)]
    samples: List[Tuple[float,float]] = []
    active: List[int] = []

    def grid_ij(p): return int(p[1]//cell), int(p[0]//cell)
    def ok(p):
        gy, gx = grid_ij(p)
        for yy in range(max(gy-2,0), min(gy+3, gh)):
            for xx in range(max(gx-2,0), min(gx+3, gw)):
                si = grid[yy][xx]
                if si == -1: continue
                qx, qy = samples[si]
                if (p[0]-qx)**2 + (p[1]-qy)**2 < r_min*r_min: return False
        return True

    p0 = (rng.uniform(0,W), rng.uniform(0,H))
    samples.append(p0); active.append(0)
    gy, gx = grid_ij(p0); grid[gy][gx] = 0
    while active:
        a = rng.choice(active)
        base = samples[a]
        found = False
        for _ in range(k):
            th = rng.uniform(0, 2*math.pi)
            rr = rng.uniform(r_min, 2*r_min)
            cand = (base[0] + rr*math.cos(th), base[1] + rr*math.sin(th))
            if 0<=cand[0]<W and 0<=cand[1]<H and ok(cand):
                samples.append(cand); active.append(len(samples)-1)
                gy, gx = grid_ij(cand); grid[gy][gx] = len(samples)-1
                found = True; break
        if not found: active.remove(a)
    return samples

def _random_polyline(canvas_size, focus_ratio, n=15, step0=40, step_sigma=0.35, rng=np.random):
    fx0, fy0, fx1, fy1 = _canvas_focus_box(canvas_size, focus_ratio)
    margin = 80
    x = rng.uniform(fx0 + margin, fx1 - margin)
    y = rng.uniform(fy0 + margin, fy1 - margin)
    pts = [(x, y)]
    ang = rng.uniform(-180, 180)
    for _ in range(1, n):
        ang += rng.normal(0.0, 40.0)
        step = max(4.0, rng.normal(step0, step0*step_sigma))
        x = min(max(x + step*math.cos(math.radians(ang)), 0.0), canvas_size[0])
        y = min(max(y + step*math.sin(math.radians(ang)), 0.0), canvas_size[1])
        pts.append((x, y))
    return pts

def _random_polygon(canvas_size, focus_ratio, n=8, rng=np.random):
    # 取焦点框内随机点 → 按极角排序近似凸多边形（避免依赖 scipy）
    fx0, fy0, fx1, fy1 = _canvas_focus_box(canvas_size, focus_ratio)
    cloud = np.stack([rng.uniform(fx0, fx1, n), rng.uniform(fy0, fy1, n)], axis=1)
    c = cloud.mean(axis=0, keepdims=True)
    ang = np.arctan2(cloud[:,1]-c[0,1], cloud[:,0]-c[0,0])
    order = np.argsort(ang)
    return cloud[order].astype(float).tolist()


def _poly_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _segments_self_intersect(P: np.ndarray) -> bool:
    M = P.shape[0]
    if M < 4:
        return False

    def seg_int(a, b, c, d):
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
            return False
        return (o1 * o2 < 0) and (o3 * o4 < 0)

    for i in range(M - 1):
        a, b = P[i], P[i + 1]
        for j in range(i + 2, M - 1):
            c, d = P[j], P[j + 1]
            if seg_int(a, b, c, d):
                return True
    return False


def _polygon_self_intersect(P: np.ndarray) -> bool:
    Q = np.vstack([P, P[0]])
    return _segments_self_intersect(Q)


def _inset_rect(W, H, margin):
    return (margin, margin, W - margin, H - margin)


def _project_to_inset(P: np.ndarray, inset: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = inset
    P[:, 0] = np.clip(P[:, 0], x0, x1)
    P[:, 1] = np.clip(P[:, 1], y0, y1)
    return P


def _min_spacing_ok(P: np.ndarray, s_min: float) -> bool:
    if P.shape[0] < 2:
        return True
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    return bool(np.all(d >= s_min - 1e-9))


def generate_polyline_by_length(
    frame_size: tuple[int, int],
    L_target: float,
    cfg: RouteGenCfg,
    rng: np.random.Generator,
) -> np.ndarray:
    W, H, D, A = _frame_metrics(frame_size)
    s_min = max(cfg.min_vertex_spacing_scale * D, 1e-6)
    margin = cfg.min_edge_margin_scale * D
    inset = _inset_rect(W, H, margin)

    for _ in range(32):
        cx = rng.uniform(inset[0], inset[2])
        cy = rng.uniform(inset[1], inset[3])
        theta = rng.uniform(0, 2 * np.pi)

        u = np.array([np.cos(theta), np.sin(theta)], float)
        p0 = np.array([cx, cy], float) - 0.5 * L_target * u
        p1 = np.array([cx, cy], float) + 0.5 * L_target * u
        P = np.stack([p0, p1], axis=0)
        P = _project_to_inset(P, np.array(inset, float))

        L_eff = float(np.linalg.norm(P[1] - P[0]))
        M = max(2, int(np.clip(round(L_eff / (0.06 * D)), 2, 64)))
        t = np.linspace(0.0, 1.0, M)
        P = (1 - t)[:, None] * P[0] + t[:, None] * P[1]

        ang_std = np.deg2rad(25.0)
        for i in range(1, M - 1):
            dvec = P[i] - P[i - 1]
            Ld = np.linalg.norm(dvec) + 1e-12
            rot = rng.normal(0.0, ang_std)
            R = np.array(
                [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], float
            )
            P[i] = P[i - 1] + (R @ (dvec / Ld)) * Ld
        P = _project_to_inset(P, np.array(inset, float))

        if not _min_spacing_ok(P, s_min):
            continue
        if _segments_self_intersect(P):
            continue
        return P.astype(np.float32)

    a = np.array([rng.uniform(inset[0], inset[2]), rng.uniform(inset[1], inset[3])])
    b = np.array([rng.uniform(inset[0], inset[2]), rng.uniform(inset[1], inset[3])])
    c = (a + b) / 2 + rng.normal(0, 0.05 * D, size=2)
    P = np.vstack([a, c, b])
    P = _project_to_inset(P, np.array(inset, float))
    return P.astype(np.float32)


def generate_polygon_by_area(
    frame_size: tuple[int, int],
    S_target: float,
    cfg: AreaGenCfg,
    rng: np.random.Generator,
) -> np.ndarray:
    W, H, D, A = _frame_metrics(frame_size)
    s_min = max(cfg.min_vertex_spacing_scale * D, 1e-6)
    margin = cfg.min_edge_margin_scale * D
    inset = _inset_rect(W, H, margin)

    for _ in range(32):
        cx = rng.uniform(inset[0], inset[2])
        cy = rng.uniform(inset[1], inset[3])
        K = int(rng.integers(12, 21))

        R0 = (S_target / np.pi) ** 0.5
        radii = np.clip(rng.normal(R0, 0.25 * R0, size=K), 0.2 * R0, 2.0 * R0)
        angs = np.sort(rng.uniform(0, 2 * np.pi, size=K))
        P = np.stack([cx + radii * np.cos(angs), cy + radii * np.sin(angs)], axis=1)

        P = _project_to_inset(P, np.array(inset, float))

        keep = [0]
        for i in range(1, K):
            if np.linalg.norm(P[i] - P[keep[-1]]) >= s_min:
                keep.append(i)
        if len(keep) < 3:
            continue
        P = P[keep]

        if _polygon_self_intersect(P):
            continue

        area = abs(_poly_area(P))
        if area <= 1e-9:
            continue
        scale = (S_target / area) ** 0.5
        P = np.column_stack([(P[:, 0] - cx) * scale + cx, (P[:, 1] - cy) * scale + cy])
        P = _project_to_inset(P, np.array(inset, float))
        area2 = abs(_poly_area(P))
        if 0.8 * S_target <= area2 <= 1.2 * S_target:
            return P.astype(np.float32)

    w = np.sqrt(S_target / 1.5)
    h = 1.5 * w
    w = np.clip(w, 0.05 * D, 0.4 * D)
    h = np.clip(h, 0.05 * D, 0.4 * D)
    cx = rng.uniform(inset[0] + w / 2, inset[2] - w / 2)
    cy = rng.uniform(inset[1] + h / 2, inset[3] - h / 2)
    P = np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ],
        float,
    )
    return P.astype(np.float32)

# =============== 文本宽度（估算） ===============
def _measure_text_width(text: str, ascii_w: int = 9, cjk_w: int = 18, max_width: int = 820) -> int:
    def is_wide(ch: str) -> bool: return ord(ch) > 127
    width = sum(cjk_w if is_wide(c) else ascii_w for c in text)
    return min(width, max_width)

def _rand_letters(n: int, rng=np.random) -> str:
    if n <= 0: return ""
    alphabet = list(string.ascii_letters)
    first = rng.choice(list(string.ascii_uppercase))
    rest = "".join(rng.choice(alphabet) for _ in range(max(0, n-1)))
    return first + rest

def _label_specs_for_len(text_len: int, rng=np.random) -> Dict[str, Dict[str, float]]:
    name = _rand_letters(text_len, rng=rng)
    w_single = _measure_text_width(name)
    extra_len = max(1, int(text_len * rng.uniform(0.5, 1.2)))
    detail = f"{name} {_rand_letters(extra_len, rng=rng)}"
    w_detail = _measure_text_width(detail)
    return {"single": {"w": float(w_single)}, "detail": {"w": float(w_detail)}}

# =============== 公开 API ===============
def generate_scene(
    canvas_size: Tuple[float, float] = _DEF_CANVAS,
    focus_ratio: float = 0.8,
    n_points: int = 8,
    n_lines: int = 2,
    n_areas: int = 2,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a random scene directly in ``SceneData`` format.

    The previous implementation returned lists and separate label metadata which
    required an adapter before feeding the solver.  This version constructs the
    arrays expected by the solver directly:

    - ``points``: ``(Np,2)`` float array.
    - ``lines``: list of ``(Mi,2)`` float arrays for polylines.
    - ``areas``: list of dicts with ``{"polygon": np.ndarray}``.
    - ``labels_init``/``WH``/``anchors``: arrays for every label (one per
      element) so the solver can be invoked without any extra conversion step.
    """

    _apply_seed(seed)
    rng = np.random.default_rng(seed)
    W, H = canvas_size

    # ---- points ---------------------------------------------------------
    pts_xy = _poisson_disc((W, H), r_min=H / 20.0, rng=rng)
    if len(pts_xy) < n_points:
        extras = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(n_points - len(pts_xy))]
        pts_xy += extras
    if n_points > 0:
        pts_xy = np.stack(pts_xy[:n_points]).astype(float)
    else:
        pts_xy = np.zeros((0, 2), float)

    # ---- lines ---------------------------------------------------------
    lines: List[np.ndarray] = []
    route_cfg = RouteGenCfg()
    for _ in range(n_lines):
        L = _sample_route_length((W, H), route_cfg, rng)
        poly = generate_polyline_by_length((W, H), L, route_cfg, rng)
        lines.append(poly.astype(float))

    # ---- areas ---------------------------------------------------------
    areas: List[Dict[str, Any]] = []
    area_cfg = AreaGenCfg()
    for _ in range(n_areas):
        S = _sample_area_size((W, H), area_cfg, rng)
        poly = generate_polygon_by_area((W, H), S, area_cfg, rng)
        areas.append({"polygon": poly.astype(float)})

    assert isinstance(lines, list) and all(
        isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 2 and p.shape[0] >= 2
        for p in lines
    )
    assert isinstance(areas, list) and all(
        isinstance(a, dict)
        and isinstance(a.get("polygon"), np.ndarray)
        and a["polygon"].ndim == 2
        and a["polygon"].shape[1] == 2
        and a["polygon"].shape[0] >= 3
        for a in areas
    )

    # ---- labels / anchors / sizes --------------------------------------
    anchors: List[np.ndarray] = []
    labels_init: List[np.ndarray] = []
    WH: List[np.ndarray] = []
    labels_meta: List[Dict[str, Any]] = []

    data_geo = {"points": pts_xy, "lines": lines, "areas": [a["polygon"] for a in areas]}

    def _rand_label_spec() -> float:
        spec = _label_specs_for_len(int(rng.integers(6, 25)), rng=rng)
        return float(spec["single"]["w"])

    # point labels
    for i in range(len(pts_xy)):
        qx, qy = anchor_xy("point", i, data_geo, (W, H))
        anc = np.array([qx, qy], float)
        anchors.append(anc)
        labels_init.append(init_position("point", anc, (W, H)))
        WH.append(np.array([_rand_label_spec(), 24.0], float))
        labels_meta.append({"id": f"p{i}", "anchor_kind": "point", "anchor_index": i})

    # line labels
    for i in range(len(lines)):
        qx, qy, meta = anchor_xy("line", i, data_geo, (W, H), with_meta=True)
        anc = np.array([qx, qy], float)
        anchors.append(anc)
        labels_init.append(init_position("line", anc, (W, H), meta=meta))
        WH.append(np.array([_rand_label_spec(), 24.0], float))
        labels_meta.append({"id": f"l{i}", "anchor_kind": "line", "anchor_index": i})

    # area labels
    for i, ar in enumerate(areas):
        qx, qy, meta = anchor_xy("area", i, data_geo, (W, H), with_meta=True)
        anc = np.array([qx, qy], float)
        anchors.append(anc)
        labels_init.append(init_position("area", anc, (W, H), meta=meta))
        WH.append(np.array([_rand_label_spec(), 24.0], float))
        labels_meta.append({"id": f"a{i}", "anchor_kind": "area", "anchor_index": i})

    scene = dict(
        frame=0,
        frame_size=(W, H),
        points=pts_xy,
        lines=lines,
        areas=areas,
        labels_init=np.vstack(labels_init),
        WH=np.vstack(WH),
        anchors=np.vstack(anchors),
        labels=labels_meta,
    )
    return scene


def make_scene_script(
    points: List[Dict[str, Any]],
    lines: List[Dict[str, Any]],
    areas: List[Dict[str, Any]],
    base_modes: Dict[str, str],
    seed: Optional[int] = None,
    n_extra: Optional[int] = None,
    allow_hide: bool = True,
    hide_weight: float = 0.25,
) -> List[Dict[str, Any]]:
    """Generate a random appear/change/hide timeline for elements.

    Parameters
    ----------
    points/lines/areas: lists of element dicts containing ``id``.
    base_modes: preferred initial mode for each element.
    seed: optional RNG seed.
    n_extra: number of ``change`` actions; defaults to ``len(elements)``.
    allow_hide: whether to allow ``hide`` actions.
    hide_weight: relative weight of hide when sampling operations.
    """
    _apply_seed(seed)
    rng = np.random.default_rng(seed)

    ids_pt = [p["id"] for p in points]
    ids_li = [l["id"] for l in lines]
    ids_ar = [a["id"] for a in areas]
    ids_all = ids_pt + ids_li + ids_ar
    N = len(ids_all)

    def allowed_modes(eid: str) -> List[str]:
        return ["single", "detail", "circle"] if eid.startswith("p") else ["single", "detail"]

    appeared: set[str] = set()
    active_modes: Dict[str, str] = {}

    tgt_extra = max(2, int(N * 1.0)) if n_extra is None else int(n_extra)
    extra_done = 0

    steps: List[Dict[str, Any]] = []
    max_steps = 10 * (N + tgt_extra) + 50
    step_count = 0

    while step_count < max_steps and (len(appeared) < N or extra_done < tgt_extra):
        step_count += 1
        not_appeared = [eid for eid in ids_all if eid not in appeared]
        have_active = len(active_modes) > 0

        ops: List[str] = []
        weights: List[float] = []

        if not_appeared:
            ops.append("appear"); weights.append(1.0)
        if have_active:
            ops.append("change"); weights.append(1.0)
        if allow_hide and len(active_modes) >= 2:
            ops.append("hide"); weights.append(hide_weight)

        if not ops:
            break

        op = rng.choice(ops, p=np.array(weights) / np.sum(weights))

        if op == "appear":
            eid = rng.choice(not_appeared)
            m0 = base_modes.get(eid, rng.choice(allowed_modes(eid)))
            active_modes[eid] = m0
            appeared.add(eid)
            steps.append({"op": "appear", "id": eid, "mode": str(m0)})

        elif op == "change":
            if not have_active:
                continue
            eid = rng.choice(list(active_modes.keys()))
            old = active_modes[eid]
            modes = [m for m in allowed_modes(eid) if m != old] or [old]
            new_mode = str(rng.choice(modes))
            active_modes[eid] = new_mode
            steps.append({"op": "change", "id": eid, "mode": new_mode})
            extra_done += 1

        elif op == "hide":
            eid = rng.choice(list(active_modes.keys()))
            del active_modes[eid]
            steps.append({"op": "hide", "id": eid})

    for eid in ids_all:
        if eid not in appeared:
            m0 = base_modes.get(eid, rng.choice(allowed_modes(eid)))
            active_modes[eid] = m0
            appeared.add(eid)
            steps.append({"op": "appear", "id": eid, "mode": str(m0)})

    for k, a in enumerate(steps):
        a["step_id"] = int(k)
    return steps

def get_scene(
    use_random: bool = True,
    cache_path: Optional[str] = None,
    with_scene_script: bool = True,
    **gen_kwargs
) -> Dict[str, Any]:
    """Generate or load a scene together with its scene script.

    ``use_random`` controls whether a new random scene is generated.  When set
    to ``False`` the function attempts to load a previously cached scene and its
    accompanying ``scene_script``.  If the cache is missing, a new scene and
    script are generated and written to ``cache_path``.  When
    ``with_scene_script`` is ``False`` any cached script is omitted from the
    returned data and no new script is generated.
    """

    if cache_path is None:
        cache_path = _default_cache_path()

    seed = gen_kwargs.get("seed", None)
    _apply_seed(seed)

    def _ensure_script(data: Dict[str, Any]) -> bool:
        if not with_scene_script:
            data.pop("scene_script", None)
            return False
        if "scene_script" in data and isinstance(data["scene_script"], list):
            return False

        n_pts = data.get("points", np.zeros((0, 2))).shape[0]
        _ln = data.get("lines")
        n_lines = len(_ln) if isinstance(_ln, list) else int(getattr(_ln, "shape", (0, 0, 0))[0])
        n_areas = len(data.get("areas", []))

        pts = [{"id": f"p{i}"} for i in range(n_pts)]
        lns = [{"id": f"l{i}"} for i in range(n_lines)]
        ars = [{"id": f"a{i}"} for i in range(n_areas)]
        modes = {lbl["id"]: "single" for lbl in data.get("labels", [])}

        data["scene_script"] = make_scene_script(
            pts, lns, ars, modes, seed=seed
        )
        return True
    if use_random:
        data = generate_scene(**gen_kwargs)
        _ensure_script(data)
        save_scene(data, cache_path)
        return data

    try:
        data = load_scene(cache_path)
    except FileNotFoundError:
        data = generate_scene(**gen_kwargs)
        _ensure_script(data)
        save_scene(data, cache_path)
        return data

    if _ensure_script(data):
        save_scene(data, cache_path)
    return data
