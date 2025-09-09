"""Spring force pulling labels toward resolved anchors."""
from __future__ import annotations

import math
import numpy as np

from cartoweave.compute.geom_anchor_resolver import anchor_position
from cartoweave.compute.geometry_sdf_smooth import (
    huber_prime,
    rect_implicit_smooth_world,
    sdf_from_implicit,
    smooth_abs,
)

from . import register, register_probe, term_cfg
from ._common import (
    read_labels_aligned,
    get_mode,
    get_ll_kernel,
    normalize_WH_from_labels,
    ensure_vec2,
    float_param,
)


@register("anchor.spring")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    """Evaluate anchor spring forces with optional ``sdf_smooth`` mode."""

    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "anchor.spring"}

    tc = term_cfg(cfg, "anchor", "spring")

    mode_default = str(tc.get("mode_spring", "center")).lower()
    k_default = float(tc.get("k_spring", 1.0))
    r0_default = float(tc.get("r0", 0.0))
    smooth_abs_eps_default = float(tc.get("smooth_abs_eps", 1e-6))
    huber_delta_default = float(tc.get("huber_delta", 0.25))
    rect_alpha_default = float(tc.get("rect_alpha", 24.0))

    N = int(P.shape[0])
    labels = read_labels_aligned(scene, P)
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask

    anchors = np.asarray(scene.get("anchors"), float)
    if anchors.shape != (N, 2):
        anchors = np.asarray([anchor_position(l, scene, P) for l in labels], float)

    WH = None

    F = np.zeros_like(P, float)
    E = 0.0
    for i in range(N):
        if not mask[i]:
            continue

        lab = labels[i]
        a = lab.get("anchor") if isinstance(lab, dict) else getattr(lab, "anchor", None)

        mode_i = mode_default
        k_i = k_default
        r0_i = r0_default
        smooth_abs_eps_i = smooth_abs_eps_default
        huber_delta_i = huber_delta_default
        rect_alpha_i = rect_alpha_default

        if a is not None:
            if hasattr(a, "get"):
                mode_i = str(a.get("mode_spring", mode_i)).lower()
                k_i = float(a.get("k_spring", k_i))
                r0_i = float(a.get("r0", r0_i))
                smooth_abs_eps_i = float(a.get("smooth_abs_eps", smooth_abs_eps_i))
                huber_delta_i = float(a.get("huber_delta", huber_delta_i))
                rect_alpha_i = float(a.get("rect_alpha", rect_alpha_i))
            else:
                mode_i = str(getattr(a, "mode_spring", mode_i)).lower()
                k_i = float(getattr(a, "k_spring", k_i))
                r0_i = float(getattr(a, "r0", r0_i))
                smooth_abs_eps_i = float(getattr(a, "smooth_abs_eps", smooth_abs_eps_i))
                huber_delta_i = float(getattr(a, "huber_delta", huber_delta_i))
                rect_alpha_i = float(getattr(a, "rect_alpha", rect_alpha_i))

        if k_i <= 0.0:
            continue

        if mode_i == "center":
            dx = float(P[i, 0] - anchors[i, 0])
            dy = float(P[i, 1] - anchors[i, 1])
            dist = math.hypot(dx, dy)
            if dist <= r0_i:
                continue
            d = dist - r0_i
            E += 0.5 * k_i * (d * d)
            scale = -k_i * d / max(dist, 1e-12)
            F[i, 0] += scale * dx
            F[i, 1] += scale * dy
            continue

        if mode_i == "sdf_smooth":
            if WH is None:
                WH = normalize_WH_from_labels(labels, N, "anchor.spring")
            llk = get_ll_kernel(lab)
            if llk not in ("rect", "ll.rect"):
                raise ValueError(
                    f"anchor.sdf_smooth: unsupported ll_kernel={llk}; only rect implemented"
                )

            C = P[i]
            e = 0.5 * WH[i]
            R = None
            R_attr = lab.get("R") if isinstance(lab, dict) else getattr(lab, "R", None)
            if R_attr is not None:
                R = np.asarray(R_attr, float).reshape(2, 2)

            A = anchors[i]
            F_imp, gradF = rect_implicit_smooth_world(
                A[None, :], C, e, R=R, alpha=rect_alpha_i
            )
            F_imp = F_imp[0]
            gradF = gradF[0]
            s, n = sdf_from_implicit(F_imp, gradF)
            L = smooth_abs(s, smooth_abs_eps_i)
            sgn = s / (L + 1e-12)
            d = L - r0_i
            if huber_delta_i > 0.0:
                d_eff = huber_prime(d, huber_delta_i)
                phi = np.sqrt(d * d + huber_delta_i * huber_delta_i)
                E += k_i * (phi - huber_delta_i)
            else:
                d_eff = d
                E += 0.5 * k_i * (d * d)
            F_i = -k_i * d_eff * sgn * n
            F[i] += F_i
            continue

        raise ValueError(f"anchor.spring: unknown mode={mode_i}")

    return float(E), ensure_vec2(F, N), {"term": "anchor.spring"}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample ``anchor.spring`` field at world coordinates ``xy``."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    anchors = np.asarray(scene.get("anchors", []), float)
    if anchors.size == 0:
        return np.zeros_like(xy, float)

    k_local = float_param(params, "k_spring", 1.0)
    if k_local <= 0.0:
        return np.zeros_like(xy, float)

    zero_dist = max(0.0, float_param(params, "r0", 0.0))

    F = np.zeros_like(xy, float)
    for ax, ay in anchors:
        dx = xy[:, 0] - float(ax)
        dy = xy[:, 1] - float(ay)
        dist = np.hypot(dx, dy)
        d = np.maximum(dist - zero_dist, 0.0)
        scale = -k_local * d / np.maximum(dist, 1e-12)
        F[:, 0] += scale * dx
        F[:, 1] += scale * dy

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("anchor.spring")(probe)

