# src/cartoweave/data/random.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Callable
import os, time, math, string, random
import numpy as np

Array = np.ndarray

# ---------- 非物理键，允许默认值 ----------
_DEF_CACHE_NAME = "scene_cache.npz"
_DEF_CANVAS = (1080.0, 1920.0)

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
    - ``lines``: ``(Nl,4)`` float array of simple segments ``(x0,y0,x1,y1)``.
    - ``areas``: list of dicts with ``{"polygon": np.ndarray}``.
    - ``labels_init``/``WH``/``anchors``: arrays for every label (one per
      element) so the solver can be invoked without any extra conversion step.
    """

    _apply_seed(seed)
    W, H = canvas_size

    # ---- points ---------------------------------------------------------
    pts_xy = _poisson_disc((W, H), r_min=H / 20.0, rng=np.random)
    if len(pts_xy) < n_points:
        extras = [
            (np.random.uniform(0, W), np.random.uniform(0, H))
            for _ in range(n_points - len(pts_xy))
        ]
        pts_xy += extras
    pts_xy = np.array(pts_xy[:n_points], float)

    # ---- lines (as simple segments) ------------------------------------
    line_segs: List[List[float]] = []
    for _ in range(n_lines):
        a = np.random.uniform(0, 1, size=2) * [W, H]
        b = np.random.uniform(0, 1, size=2) * [W, H]
        line_segs.append([float(a[0]), float(a[1]), float(b[0]), float(b[1])])
    lines = np.array(line_segs, float).reshape(-1, 4)

    # ---- areas ---------------------------------------------------------
    areas: List[Dict[str, Any]] = []
    for _ in range(n_areas):
        poly = np.array(
            _random_polygon((W, H), focus_ratio, n=int(np.random.randint(6, 11)), rng=np.random),
            float,
        )
        areas.append({"polygon": poly})

    # ---- labels / anchors / sizes --------------------------------------
    anchors: List[np.ndarray] = []
    labels_init: List[np.ndarray] = []
    WH: List[np.ndarray] = []
    labels_meta: List[Dict[str, Any]] = []

    def _rand_label_spec() -> float:
        spec = _label_specs_for_len(int(np.random.randint(6, 25)))
        return float(spec["single"]["w"])

    # point labels
    for i, (x, y) in enumerate(pts_xy):
        anc = np.array([x, y], float)
        anchors.append(anc)
        offset = np.random.uniform(-20.0, 20.0, size=2)
        labels_init.append(anc + offset)
        WH.append(np.array([_rand_label_spec(), 24.0], float))
        labels_meta.append({"id": f"p{i}", "anchor_kind": "point", "anchor_index": i})

    # line labels
    for i, seg in enumerate(lines):
        anc = np.array([(seg[0] + seg[2]) * 0.5, (seg[1] + seg[3]) * 0.5], float)
        anchors.append(anc)
        offset = np.random.uniform(-20.0, 20.0, size=2)
        labels_init.append(anc + offset)
        WH.append(np.array([_rand_label_spec(), 24.0], float))
        labels_meta.append({"id": f"l{i}", "anchor_kind": "line", "anchor_index": i})

    # area labels
    for i, ar in enumerate(areas):
        poly = np.asarray(ar["polygon"], float)
        anc = poly.mean(axis=0)
        anchors.append(anc)
        offset = np.random.uniform(-20.0, 20.0, size=2)
        labels_init.append(anc + offset)
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


def make_timeline(
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

    actions: List[Dict[str, Any]] = []
    max_steps = 10 * (N + tgt_extra) + 50
    steps = 0

    while steps < max_steps and (len(appeared) < N or extra_done < tgt_extra):
        steps += 1
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
            actions.append({"op": "appear", "id": eid, "mode": str(m0)})

        elif op == "change":
            if not have_active:
                continue
            eid = rng.choice(list(active_modes.keys()))
            old = active_modes[eid]
            modes = [m for m in allowed_modes(eid) if m != old] or [old]
            new_mode = str(rng.choice(modes))
            active_modes[eid] = new_mode
            actions.append({"op": "change", "id": eid, "mode": new_mode})
            extra_done += 1

        elif op == "hide":
            eid = rng.choice(list(active_modes.keys()))
            del active_modes[eid]
            actions.append({"op": "hide", "id": eid})

    for eid in ids_all:
        if eid not in appeared:
            m0 = base_modes.get(eid, rng.choice(allowed_modes(eid)))
            active_modes[eid] = m0
            appeared.add(eid)
            actions.append({"op": "appear", "id": eid, "mode": str(m0)})

    for k, a in enumerate(actions):
        a["aid"] = int(k)
    return actions

def get_scene(
    use_random: bool = True,
    cache_path: Optional[str] = None,
    *,
    with_timeline: bool = False,
    auto_make_timeline_if_missing: bool = False,
    make_timeline_fn: Optional[Callable[[Dict[str, Any]], List[dict]]] = None,
    timeline_kwargs: Optional[Dict[str, Any]] = None,
    **gen_kwargs
) -> Dict[str, Any]:
    """
    行为与旧工程一致：
    - use_random=True  : 重新生成（可带 seed）→ （可选）生成 timeline → 写缓存 → 返回
    - use_random=False : 读缓存；若无则先生成并写缓存；（可选）缺失 timeline 时补齐并回写
    当 ``make_timeline_fn`` 为 ``None`` 且 ``with_timeline`` 为 ``True`` 时，默认使
    用 :func:`make_timeline`。
    """
    if cache_path is None:
        cache_path = _default_cache_path()

    _apply_seed(gen_kwargs.get("seed", None))

    if use_random:
        data = generate_scene(**gen_kwargs)
        if with_timeline and callable(make_timeline_fn):
            tl = make_timeline_fn(dict(data), **(timeline_kwargs or {}))
            data["timeline"] = tl
        save_scene(data, cache_path)
        return data

    try:
        data = load_scene(cache_path)
    except FileNotFoundError:
        data = generate_scene(**gen_kwargs)
        save_scene(data, cache_path)

    if with_timeline and ("timeline" not in data or not isinstance(data["timeline"], list) or len(data["timeline"]) == 0):
        if auto_make_timeline_if_missing and callable(make_timeline_fn):
            tl = make_timeline_fn(dict(data), **(timeline_kwargs or {}))
            data["timeline"] = tl
            save_scene(data, cache_path)
    return data
