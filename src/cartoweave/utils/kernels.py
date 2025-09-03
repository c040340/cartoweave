from __future__ import annotations
import math
import numpy as np

# 统一 eps（cfg 里可覆盖，没给就用兜底）
EPS_NORM = 1e-9    # 单位向量防 0
EPS_DIST = 1e-6    # 距离防 0
EPS_ABS  = 1e-3    # softabs 平滑

def safe_hypot(x: float, y: float, eps: float = EPS_NORM) -> float:
    return math.hypot(x, y) + eps

def softabs(x: float, eps: float = EPS_ABS) -> float:
    return math.sqrt(x*x + eps*eps)

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1.0/(1.0+z)
    else:
        z = math.exp(x);  return z/(1.0+z)

def softplus(x: float, beta: float) -> float:
    b = max(beta, 1e-9)
    t = b * x
    if t > 30:  return x
    if t < -30: return math.exp(t) / b
    return math.log1p(math.exp(t)) / b

def logcosh_energy(x: float, p0: float) -> float:
    t = x / max(p0, 1e-9)
    return max(p0, 1e-9) * (abs(t) + math.log1p(math.exp(-2.0*abs(t))) - math.log(2.0))

def d_logcosh(x: float, p0: float) -> float:
    return math.tanh(x / max(p0, 1e-9))

def invdist_energy(c: float, k: float, p: float) -> float:
    cc = max(c, 1e-12)
    if p == 1.0:
        return k * math.log(cc)
    return k * (cc ** (1.0 - p)) / (p - 1.0)

def invdist_force_mag(c: float, k: float, p: float) -> float:
    return k / (max(c, 1e-12) ** p)

def softmin_weights(vals: np.ndarray, beta: float) -> np.ndarray:
    v = np.asarray(vals, float)
    b = max(beta, 1e-9)
    w = np.exp(-b * (v - v.min()))
    s = w.sum() + 1e-12
    return w / s
