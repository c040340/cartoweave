from __future__ import annotations
import numpy as np


def read_labels_aligned(scene: dict, P: np.ndarray):
    labels = scene.get("labels") or []
    if len(labels) != int(P.shape[0]):
        raise ValueError(f"[index] labels len {len(labels)} != P rows {P.shape[0]}")
    return labels


def get_mode(lab, default: str = "") -> str:
    # real class / behavior role
    if isinstance(lab, dict):
        m = lab.get("mode", None)
        if m is None:
            m = (lab.get("meta") or {}).get("mode", default)
        return (m or default) if isinstance(m, str) else default
    m = getattr(lab, "mode", None)
    if m is None:
        m = (getattr(lab, "meta", None) or {}).get("mode", default)
    return (m or default) if isinstance(m, str) else default


def get_ll_kernel(lab, default: str | None = None) -> str | None:
    # label-label kernel selector: "rect" | "disk" (ellipse treated as disk)
    if isinstance(lab, dict):
        v = lab.get("ll_kernel", default)
    else:
        v = getattr(lab, "ll_kernel", default)
    if v is None:
        return None
    v = str(v).lower()
    if v in ("ellipse", "elliptic", "circle"):  # canonicalize synonyms
        return "disk"
    if v in ("rectangle",):
        return "rect"
    return v


def normalize_WH_from_labels(labels, N: int, log_name: str) -> np.ndarray:
    # Strict: raise on bad shape. No silent fallback.
    arr = []
    for lab in labels:
        if isinstance(lab, dict):
            wh = lab.get("WH", None)
        else:
            wh = getattr(lab, "WH", None)
        if wh is None:
            raise ValueError(f"[{log_name}] missing WH on a label")
        a = np.asarray(wh, float)
        arr.append(a)
    WH = np.asarray(arr, float)
    if WH.ndim == 1:
        if WH.shape[0] == 2:
            WH = np.broadcast_to(WH.reshape(1, 2), (N, 2)).astype(float, copy=False)
        elif WH.shape[0] == N:
            WH = np.stack([WH, WH], axis=1).astype(float, copy=False)
        else:
            raise ValueError(f"[{log_name}] WH 1D shape {WH.shape} not compatible with N={N}")
    if WH.shape != (N, 2):
        raise ValueError(f"[{log_name}] WH shape {WH.shape} != (N,2)")
    return WH


def ensure_vec2(F: np.ndarray, N: int) -> np.ndarray:
    F = np.asarray(F, float)
    if F.shape != (N, 2):
        raise ValueError(f"force shape {F.shape} != (N,2)")
    return F


def float_param(d: dict | None, key: str, default: float) -> float:
    """Safely extract a finite float from ``d``.

    ``params`` dictionaries coming from the UI may contain ``None`` or even
    nested dictionaries when sliders are disabled.  Directly calling
    ``float()`` on such values raises ``TypeError`` and would bubble up as an
    exception from probe functions.  This helper gracefully falls back to a
    provided default and also guards against non‑finite values like ``NaN`` or
    ``±∞``.
    """

    if not isinstance(d, dict):  # malformed params container
        v = default
    else:
        v = d.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = default
    if not np.isfinite(v):
        v = default
    return float(v)


def poly_as_array(poly) -> np.ndarray | None:
    """Return an ``(N,2)`` float array from a polygon-like input.

    ``scene['areas']`` can contain bare coordinate lists or dictionaries
    with a ``'poly'`` field.  Probes should skip malformed polygons rather
    than raise ``TypeError``/``ValueError`` when visualizations toggle force
    terms on and off.  This helper extracts the vertex array while guarding
    against bad shapes and non-finite coordinates.
    """

    if isinstance(poly, dict):
        poly = poly.get("poly")
    if poly is None:
        return None
    arr = np.asarray(poly, float)
    try:
        arr = arr.reshape(-1, 2)
    except Exception:
        return None
    if arr.shape[0] < 3:
        return None
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

