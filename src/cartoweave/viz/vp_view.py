from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Sequence, Tuple
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from cartoweave.contracts.viewpack_v1 import ViewPack, VPFrame, VPPass

logger = logging.getLogger("cartoweave.viz.vp_view")

# ---------------------- 轻量配置 ----------------------
@dataclass
class VizOpts:
    show_sources: bool = True
    show_rects: bool = True
    show_forces: bool = True
    force_scale: float = 1.0
    only_active: bool = True
    terms: Optional[Sequence[str]] = None
    annotate: bool = True

# ---------------------- 从 ViewPack 构建场景 ----------------------
@dataclass
class VPScene:
    W: float
    H: float
    traj: np.ndarray
    WH: np.ndarray
    points: np.ndarray
    lines: Sequence[np.ndarray]
    areas: Sequence[Sequence[Tuple[float,float]]]
    passes: Sequence[VPPass]
    labels: Sequence[Dict[str, Any]]

def build_scene(vp: ViewPack) -> VPScene:
    T = vp.num_frames()
    N = vp.N
    traj = np.stack([fr.P for fr in vp.frames], axis=0)
    WH = np.asarray(vp.WH, float) if (vp.WH is not None) else np.zeros((N, 2), float)
    src = vp.sources
    points = np.asarray(src.points, float) if len(src.points) else np.zeros((0,2), float)
    lines = [np.asarray(ln, float) for ln in src.lines] if len(src.lines) else []
    areas = list(src.areas) if len(src.areas) else []
    W, H = map(float, src.frame_size)
    return VPScene(
        W=W, H=H, traj=traj, WH=WH, points=points, lines=lines, areas=areas,
        passes=tuple(sorted(vp.passes, key=lambda p:(p.t_start,p.t_end))),
        labels=vp.labels
    )

# ---------------------- passes 索引（一次性预处理） ----------------------
def _index_passes(vp: ViewPack):
    T = vp.num_frames()
    if vp.passes:
        ps = sorted(vp.passes, key=lambda p: (p.t_start, p.t_end))
        frame_to_pass = np.full(T, -1, int)
        starts = np.array([int(p.t_start) for p in ps], int)
        ends   = np.array([int(p.t_end)   for p in ps], int)
        for i,(a,b) in enumerate(zip(starts, ends)):
            a = max(0, min(T, a)); b = max(0, min(T, b))
            if a < b: frame_to_pass[a:b] = i
        iter_in_pass = np.empty(T, dtype=object)
        for t in range(T):
            k = frame_to_pass[t]
            iter_in_pass[t] = (t - starts[k]) if k >= 0 else None
        boundaries = list(starts) + [max(ends.max(initial=0), T)]
        return tuple(ps), frame_to_pass, iter_in_pass, boundaries
    # 回退：用 meta
    frame_to_pass = np.full(T, -1, int)
    starts = []
    last = -1
    iter_in_pass = np.empty(T, dtype=object)
    for t, fr in enumerate(vp.frames):
        k = fr.meta.get("pass_id", None)
        if isinstance(k, int):
            frame_to_pass[t] = k
            if k != last:
                starts.append(t); last = k
        iter_in_pass[t] = fr.meta.get("frame_in_pass", None)
    boundaries = starts + [T]
    pseudo = []
    for i, a in enumerate(starts or [0]):
        b = boundaries[i+1] if i+1 < len(boundaries) else T
        pseudo.append(type("VPPassPseudo",(object,),{"pass_name": f"pass-{i}","t_start":a,"t_end":b})())
    return tuple(pseudo), frame_to_pass, iter_in_pass, boundaries

# ---------------------- 绘制工具 ----------------------
def _rect_corners(center_xy: np.ndarray, wh: np.ndarray) -> np.ndarray:
    hw = 0.5*wh
    x,y = center_xy[:,0], center_xy[:,1]
    w,h = hw[:,0], hw[:,1]
    return np.stack([
        np.stack([x - w, y - h], axis=-1),
        np.stack([x - w, y + h], axis=-1),
        np.stack([x + w, y + h], axis=-1),
        np.stack([x + w, y - h], axis=-1),
    ], axis=1)

def _draw_sources(ax, sc: VPScene, opts: VizOpts):
    if not opts.show_sources: return
    if sc.points.size:
        ax.scatter(sc.points[:,0], sc.points[:,1], s=10, alpha=0.6, linewidths=0)
    if sc.lines:
        segs = [ln for ln in sc.lines if len(ln)]
        if segs:
            ax.add_collection(LineCollection(segs, linewidths=1.0, alpha=0.6))
    if sc.areas:
        polys = []
        for area in sc.areas:
            if isinstance(area, dict):
                kind = area.get("kind")
                if kind == "poly":
                    polys.append(np.asarray(area.get("xy"), float))
                elif kind == "circle":
                    xy = np.asarray(area.get("xy"), float)
                    r = float(area.get("r", 0.0))
                    th = np.linspace(0, 2*np.pi, 32)
                    poly = np.stack([xy[0] + r*np.cos(th), xy[1] + r*np.sin(th)], axis=1)
                    polys.append(poly)
            else:
                polys.append(np.asarray(area, float))
        polys = [p for p in polys if len(p)]
        if polys:
            ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="k", linewidths=0.8, alpha=0.5))

def _draw_rects(ax, P: np.ndarray, WH: np.ndarray, active_mask: np.ndarray, opts: VizOpts):
    if not opts.show_rects: return
    N = P.shape[0]
    mask = active_mask if opts.only_active else np.ones(N, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0: return
    corners = _rect_corners(P[idx], WH[idx])
    ax.add_collection(PolyCollection(corners, facecolors="none", edgecolors="tab:blue", linewidths=1.2, alpha=0.9))
    ax.scatter(P[idx,0], P[idx,1], s=5, alpha=0.8, linewidths=0)

def _draw_forces(ax, fr: VPFrame, opts: VizOpts):
    if not opts.show_forces: return
    comps = fr.comps or {}
    terms = list(comps.keys()) if opts.terms is None else [t for t in opts.terms if t in comps]
    if not terms: return
    P = fr.P
    am = fr.active_mask
    idx = np.flatnonzero(am) if opts.only_active else np.arange(P.shape[0])
    if idx.size == 0: return
    for t in terms:
        F = comps[t]
        U = F[idx] * float(opts.force_scale)
        X = P[idx]
        ax.quiver(X[:,0], X[:,1], U[:,0], U[:,1], angles="xy", scale_units="xy", scale=1.0, width=0.002, alpha=0.9)

def _annot(ax, vp: ViewPack, t: int, apass: Optional[VPPass], fr: VPFrame):
    if not fr.metrics:
        E = float(fr.E); g = fr.meta.get("gnorm", None)
    else:
        E = float(fr.metrics.get("E", fr.E))
        g = fr.metrics.get("gnorm", fr.metrics.get("g_norm", None))
    lines = [f"t={t}  E={E:.6g}"]
    if g is not None:
        try: lines.append(f"‖G‖={float(g):.3g}")
        except: pass
    if apass:
        lines.append(f"action={apass.pass_name} [{apass.t_start},{apass.t_end})")
    fi = fr.meta.get("frame_in_pass", None)
    if fi is not None: lines.append(f"in-pass={fi}")
    ax.text(0.01, 0.99, "  ".join(lines), ha="left", va="top", transform=ax.transAxes, fontsize=10,
            bbox=dict(fc="w", ec="none", alpha=0.6))

# ---------------------- 交互式浏览 ----------------------
def show_vp(vp: ViewPack, opts: Optional[VizOpts]=None):
    opts = opts or VizOpts()
    sc = build_scene(vp)
    T, N = sc.traj.shape[:2]
    logger.info("ViewPack: T=%d frames, N=%d labels", T, N)

    passes, frame_to_pass, iter_in_pass, boundaries = _index_passes(vp)

    # 轻量一致性校验：meta.pass_id 与 passes 区段是否契合
    try:
        mismatch = 0
        for t, fr in enumerate(vp.frames):
            k_meta = fr.meta.get("pass_id", None)
            k_map  = int(frame_to_pass[t]) if 0 <= t < len(frame_to_pass) else None
            if isinstance(k_meta, int) and k_map is not None and k_meta != k_map:
                mismatch += 1
        if mismatch:
            logger.warning("viz: %d frames have meta.pass_id mismatching VPPass ranges", mismatch)
    except Exception:
        pass

    def current_pass(t: int):
        k = int(frame_to_pass[t]) if 0 <= t < len(frame_to_pass) else -1
        return passes[k] if 0 <= k < len(passes) else None

    def next_action_start(t: int):
        if not boundaries: return t
        for b in boundaries:
            if b > t: return min(T-1, b)
        return T-1

    def prev_action_start(t: int):
        if not boundaries: return 0
        prev = 0
        for b in boundaries:
            if b >= t: break
            prev = b
        return prev

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(0, sc.W); ax.set_ylim(0, sc.H)

    state = {"t": 0}
    def redraw():
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(0, sc.W); ax.set_ylim(0, sc.H)
        t = state["t"]
        fr = vp.frames[t]
        _draw_sources(ax, sc, opts)
        _draw_rects(ax, sc.traj[t], sc.WH, fr.active_mask, opts)
        _draw_forces(ax, fr, opts)
        if opts.annotate:
            _annot(ax, vp, t, current_pass(t), fr)
        fig.canvas.draw_idle()

    def on_key(e):
        t = state["t"]
        if e.key in ("right", "n"):
            state["t"] = min(T-1, t+1)
        elif e.key in ("left", "p"):
            state["t"] = max(0, t-1)
        elif e.key == "a":
            state["t"] = next_action_start(t)
        elif e.key == "A":
            state["t"] = prev_action_start(t)
        elif e.key == "f":
            opts.show_forces = not opts.show_forces
        elif e.key == "r":
            opts.show_rects = not opts.show_rects
        elif e.key == "s":
            opts.show_sources = not opts.show_sources
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()

# ---------------------- 静态帧导出 ----------------------
def render_frame(vp: ViewPack, t: int, opts: Optional[VizOpts]=None, dpi: int = 180):
    opts = opts or VizOpts()
    sc = build_scene(vp)
    T = sc.traj.shape[0]
    t = int(np.clip(t, 0, T-1))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.set_aspect("equal")
    ax.set_xlim(0, sc.W); ax.set_ylim(0, sc.H)
    fr = vp.frames[t]
    _draw_sources(ax, sc, opts)
    _draw_rects(ax, sc.traj[t], sc.WH, fr.active_mask, opts)
    _draw_forces(ax, fr, opts)
    _annot(ax, vp, t, None, fr)
    return fig
