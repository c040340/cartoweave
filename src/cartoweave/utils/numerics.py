import numpy as np
from typing import Sequence


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable vectorized logistic σ(x)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = (x >= 0)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def d_sigmoid_np(x: np.ndarray) -> np.ndarray:
    """σ'(x) = σ(x)(1-σ(x)) computed stably."""
    s = sigmoid_np(x)
    return s * (1.0 - s)


def softplus_np(z: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Stable softplus with thresholding."""
    z = np.asarray(z, dtype=float)
    bz = beta * z
    out = np.empty_like(bz)
    hi = (bz > 50.0)
    lo = (bz < -50.0)
    mid = ~(hi | lo)
    out[hi] = z[hi]
    out[lo] = np.exp(bz[lo]) / beta
    out[mid] = np.log1p(np.exp(bz[mid])) / beta
    return out


def softabs_np(s: np.ndarray, eps_abs: float) -> np.ndarray:
    """Stable soft |s|; hypot is more robust than manual sqrt."""
    return np.hypot(s, eps_abs)


def softmin_weights_np(vals: np.ndarray, beta: float, use_abs: bool = False) -> np.ndarray:
    """Softmin weights with optional absolute values."""
    v = np.abs(vals) if use_abs else np.asarray(vals, float)
    v = np.asarray(v, float)
    b = max(beta, 1e-9)
    z = -b * (v - v.min())
    z = z - z.max()
    w = np.exp(z)
    return w / (w.sum() + 1e-12)


def softmin_weights_with_grad(vals: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Softmin weights and their contribution to the weighted sum gradient.

    Parameters
    ----------
    vals:
        ``(n,)`` array of values ``m_j``.
    beta:
        Softmin sharpness ``β``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(w, g)`` where ``w`` are the softmin weights and ``g_j`` are the
        coefficients ``w_j * [1 - β (m_j - m_eff)]`` used when differentiating
        the weighted sum ``m_eff = Σ w_j m_j``.
    """

    v = np.asarray(vals, float)
    b = max(beta, 1e-9)
    z = -b * (v - v.min())
    z = z - z.max()
    w = np.exp(z)
    w = w / (w.sum() + 1e-12)
    m_eff = float((w * v).sum())
    g = w * (1.0 - b * (v - m_eff))
    return w, g


def is_finite_array(x: np.ndarray) -> bool:
    """Return True if *x* contains only finite values."""
    return np.isfinite(np.asarray(x, dtype=float)).all()


def sanitize_array(x: np.ndarray) -> np.ndarray:
    """Replace non-finite values in *x* with zeros."""
    arr = np.asarray(x, dtype=float)
    if np.isfinite(arr).all():
        return arr
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def repair_nonfinite_rows(
    arr: np.ndarray, reference: np.ndarray, indices: Sequence[int] | None = None
) -> tuple[np.ndarray, list[int]]:
    """Replace non-finite rows in ``arr`` with rows from ``reference``.

    Parameters
    ----------
    arr:
        ``(M, 2)`` array to be checked in-place.
    reference:
        ``(N, 2)`` array providing fallback values.
    indices:
        Optional mapping of row indices in ``arr`` to rows in ``reference``.

    Returns
    -------
    tuple[np.ndarray, list[int]]
        The possibly modified ``arr`` and a list of the repaired reference
        indices.
    """

    a = np.asarray(arr, float)
    ref = np.asarray(reference, float)
    if a.ndim != 2 or a.shape[1] != 2:
        return a, []
    bad = ~np.isfinite(a).all(axis=1)
    if not np.any(bad):
        return a, []
    repaired: list[int] = []
    idxs = list(range(len(a))) if indices is None else list(indices)
    for local_i in np.where(bad)[0]:
        if local_i < len(idxs):
            ref_i = idxs[local_i]
            if 0 <= ref_i < len(ref):
                a[local_i] = ref[ref_i]
                repaired.append(ref_i)
    return a, repaired

