from __future__ import annotations
import math
import numpy as np

# 统一 eps（cfg 里可覆盖，没给就用兜底）
EPS_NORM = 1e-9    # 单位向量防 0
EPS_DIST = 1e-6    # 距离防 0
EPS_ABS  = 1e-3    # softabs 平滑
EPS_AREA = 1e-9    # 多边形面积防 0

def safe_hypot(x: float, y: float, eps: float = EPS_NORM) -> float:
    return math.hypot(x, y) + eps

def softabs(x: float, eps: float = EPS_ABS) -> float:
    return math.sqrt(x*x + eps*eps)

def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid supporting scalars or arrays."""
    z = np.asarray(x, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    if out.shape == ():
        return float(out)
    return out

def softplus(x: float | np.ndarray, beta: float) -> float | np.ndarray:
    """Stable softplus supporting scalars or numpy arrays."""
    b = max(beta, 1e-9)
    z = b * np.asarray(x, dtype=float)
    out = np.empty_like(z, dtype=float)
    hi = z > 30.0
    lo = z < -30.0
    mid = ~(hi | lo)
    out[hi] = z[hi] / b
    out[lo] = np.exp(z[lo]) / b
    out[mid] = np.log1p(np.exp(z[mid])) / b
    if out.shape == ():
        return float(out)  # return python float for scalar inputs
    return out

def logcosh_energy(x: float, p0: float) -> float:
    t = x / max(p0, 1e-9)
    return max(p0, 1e-9) * (abs(t) + math.log1p(math.exp(-2.0*abs(t))) - math.log(2.0))

def d_logcosh(x: float, p0: float) -> float:
    return math.tanh(x / max(p0, 1e-9))

def invdist_energy(c: float | np.ndarray, k: float, p: float) -> float | np.ndarray:
    cc = np.maximum(np.asarray(c, dtype=float), 1e-12)
    if p == 1.0:
        out = k * np.log(cc)
    else:
        out = k * (cc ** (1.0 - p)) / (p - 1.0)
    if out.shape == ():
        return float(out)
    return out

def invdist_force_mag(c: float | np.ndarray, k: float, p: float) -> float | np.ndarray:
    cc = np.maximum(np.asarray(c, dtype=float), 1e-12)
    out = k / (cc ** p)
    if out.shape == ():
        return float(out)
    return out

def softmin_weights(vals: np.ndarray, beta: float) -> np.ndarray:
    v = np.asarray(vals, float)
    b = max(beta, 1e-9)
    w = np.exp(-b * (v - v.min()))
    s = w.sum() + 1e-12
    return w / s


def smoothmax(a, b, beta=8.0):
    """
    C^∞ approx of max(a,b): (1/beta)*log(exp(beta*a)+exp(beta*b)).
    Stable for large inputs by subtracting max.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.maximum(a, b)
    return m + (np.log(np.exp(beta * (a - m)) + np.exp(beta * (b - m)))) / beta


def smoothmin(a, b, beta=8.0):
    # min(a,b) = -smoothmax(-a,-b)
    return -smoothmax(-a, -b, beta=beta)


def softclip(x, lo, hi, beta=8.0):
    """
    C^1 soft clip to [lo, hi] using two softplus walls.
    For beta→∞ approximates hard clip. beta≈6-12 works well.
    """
    x = np.asarray(x, float)
    x_hi = hi - softplus(hi - x, beta) / beta
    return lo + softplus(x_hi - lo, beta) / beta


def safe_div(num, den, eps=1e-12):
    """
    Safe division with soft lower bound on denominator using softplus.
    Keeps gradient smooth even when den≈0.
    """
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    den_safe = eps + softplus(den - eps, beta=8.0) / 8.0
    return num / den_safe
