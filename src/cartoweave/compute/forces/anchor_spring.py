"""Anchor spring force for point/line labels.

This implementation follows the mechanics described in the user provided
reference: the spring acts along the direction from the anchor to the label
centre and uses the incident distance ``s_hit`` against a target gap ``r0``.
An additional quadratic penalty pushes labels out when the anchor has already
penetrated the label rectangle (``s_hit < 0``).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from cartoweave.utils.kernels import softabs

from . import register, register_probe, term_cfg, eps_params
from ._common import (
    read_labels_aligned,
    normalize_WH_from_labels,
    ensure_vec2,
    anchor_info,
    float_param,
)


def _anchor_xy(lab, points, lines, areas) -> Tuple[str, int, float | None, float | None]:
    """Resolve anchor kind/index and coordinates for ``lab``."""

    info = anchor_info(lab)
    if not info:
        return "none", -1, None, None

    kind = info.get("kind")
    try:
        idx = int(info.get("index"))
    except (TypeError, ValueError):
        idx = -1

    if kind == "point":
        pts = np.asarray(points, float)
        if 0 <= idx < len(pts):
            ax, ay = float(pts[idx, 0]), float(pts[idx, 1])
            return kind, idx, ax, ay
        return kind, idx, None, None

    if kind == "line":
        if 0 <= idx < len(lines):
            coords = np.asarray(lines[idx], float)
            if coords.size >= 2:
                ax = float(0.5 * (coords[0, 0] + coords[-1, 0]))
                ay = float(0.5 * (coords[0, 1] + coords[-1, 1]))
                return kind, idx, ax, ay
        return kind, idx, None, None

    if kind == "area":
        if 0 <= idx < len(areas):
            poly = np.asarray(areas[idx], float)
            mins = poly.min(axis=0)
            maxs = poly.max(axis=0)
            ax = float(0.5 * (mins[0] + maxs[0]))
            ay = float(0.5 * (mins[1] + maxs[1]))
            return kind, idx, ax, ay
        return kind, idx, None, None

    return "none", -1, None, None


def anchor_spring_eval(sd, r0, k, *, kind="linear", alpha=3.0):
    """Evaluate spring energy and derivative w.r.t ``sd``.

    Parameters
    ----------
    sd: float
        Signed incident distance.
    r0: float
        Preferred clearance distance.
    k: float
        Spring stiffness.
    kind: str
        ``"linear"`` or ``"logcosh"``.
    alpha: float
        Parameter for ``logcosh`` model (acts as ``p0``).
    """

    x = float(sd - r0)
    if kind == "logcosh":
        p0 = max(1e-9, float(alpha))
        z = x / p0
        if abs(z) < 20.0:
            E = k * p0 * math.log(math.cosh(z))
        else:
            E = k * p0 * (abs(z) - math.log(2.0))
        dE_dsd = k * math.tanh(z)
        return E, dE_dsd

    E = 0.5 * k * x * x
    dE_dsd = k * x
    return E, dE_dsd


@register("anchor.spring")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    """Evaluate anchor spring forces."""

    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "anchor.spring"}

    tc = term_cfg(cfg, "anchor", "spring")
    epss = eps_params(cfg, tc, defaults={"abs": 1e-6})
    eps_abs = float(epss["eps_abs"])

    def _g(key: str, default):
        val = tc.get(key, default)
        return default if val is None else val

    k_spr = float(_g("k_anchor_spring", 1.0))
    k_occl = float(_g("k_anchor_occlusion", 0.0))
    kind = str(_g("anchor_spring_kind", "linear"))
    alpha = float(_g("anchor_spring_alpha", 3.0))
    r0_point = float(_g("anchor_r0_points", 0.0))
    r0_line = float(_g("anchor_r0_lines", 0.0))
    eps_nrm = float(_g("eps_norm", 1e-9))

    labels = read_labels_aligned(scene, P)
    N = int(P.shape[0])
    WH = normalize_WH_from_labels(labels, N, "anchor.spring")

    points = scene.get("points", []) or []
    lines = scene.get("lines", []) or []
    areas = scene.get("areas", []) or []

    F = np.zeros_like(P, float)
    E = 0.0

    for i, lab in enumerate(labels):
        ak, aidx, ax, ay = _anchor_xy(lab, points, lines, areas)

        if ak == "area" or ax is None:
            continue

        r0 = r0_line if ak == "line" else r0_point

        w, h = float(WH[i, 0]), float(WH[i, 1])
        if w <= 0.0 and h <= 0.0:
            continue

        cx, cy = float(P[i, 0]), float(P[i, 1])
        ux = cx - ax
        uy = cy - ay
        r = math.hypot(ux, uy)

        hx, hy = 0.5 * w, 0.5 * h

        if r <= eps_nrm:
            # near‑zero radius: fall back to external direction if available
            ext = scene.get("_ext_dir", None)
            if ext is not None:
                ex, ey = float(ext[i, 0]), float(ext[i, 1])
                en = math.hypot(ex, ey)
                if en > 1e-12:
                    uxh, uyh = ex / en, ey / en
                else:
                    uxh = uyh = 0.70710678
            else:
                uxh = uyh = 0.70710678
            rho = hx * softabs(uxh, eps_abs) + hy * softabs(uyh, eps_abs)
            ds_du = (uxh, uyh)
        else:
            uxh, uyh = ux / r, uy / r
            eps2 = eps_abs * eps_abs
            A = math.sqrt(ux * ux + eps2 * r * r)
            B = math.sqrt(uy * uy + eps2 * r * r)
            C = hx * A + hy * B
            rho = C / r
            dC_dux = hx * ((1.0 + eps2) * ux / A) + hy * (eps2 * ux / B)
            dC_duy = hx * (eps2 * uy / A) + hy * ((1.0 + eps2) * uy / B)
            ds_dux = ux / r - dC_dux / r + C * ux / (r * r * r)
            ds_duy = uy / r - dC_duy / r + C * uy / (r * r * r)
            ds_du = (ds_dux, ds_duy)

        s_hit = r - rho
        E_spr, dE_dsd = anchor_spring_eval(s_hit, r0, k_spr, kind=kind, alpha=alpha)
        dE_dsd_tot = dE_dsd
        if s_hit < 0.0 and k_occl > 0.0:
            E += 0.5 * k_occl * (s_hit * s_hit)
            dE_dsd_tot += k_occl * s_hit
        F[i, 0] += -(dE_dsd_tot) * ds_du[0]
        F[i, 1] += -(dE_dsd_tot) * ds_du[1]
        E += E_spr

    return float(E), ensure_vec2(F, N), {"term": "anchor.spring"}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample ``anchor.spring`` field at world coordinates ``xy``.

    This probe uses a simplified radial spring model ignoring occlusion and
    rectangle geometry.  It is primarily intended for visualization.
    """

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    anchors = np.asarray(scene.get("anchors", []), float)
    if anchors.size == 0:
        return np.zeros_like(xy, float)

    k_local = float_param(params, "k_anchor_spring", 1.0)
    if k_local <= 0.0:
        return np.zeros_like(xy, float)

    r0 = float_param(params, "anchor_r0_points", 0.0)

    F = np.zeros_like(xy, float)
    for ax, ay in anchors:
        dx = xy[:, 0] - float(ax)
        dy = xy[:, 1] - float(ay)
        dist = np.hypot(dx, dy)
        d = np.maximum(dist - r0, 0.0)
        scale = -k_local * d / np.maximum(dist, 1e-12)
        F[:, 0] += scale * dx
        F[:, 1] += scale * dy

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("anchor.spring")(probe)


__all__ = ["evaluate", "probe"]

