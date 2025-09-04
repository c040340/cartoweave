import numpy as np


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

