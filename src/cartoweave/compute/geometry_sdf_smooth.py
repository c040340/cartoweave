import numpy as np

EPS = 1e-12

def smooth_abs(x, eps: float = 1e-6):
    """C¹ 平滑 |x|，避免在 0 处不可导：sqrt(x^2 + eps^2)."""
    return np.sqrt(np.square(x) + eps * eps)

def huber_prime(d, delta: float):
    """Huber 的一阶导 φ'(d) = d / sqrt(d^2 + δ^2)，用于平滑 (L - r0)。"""
    return d / np.sqrt(np.square(d) + delta * delta + EPS)

def softmax(vals, alpha: float):
    """C¹ 光滑 max；alpha 越大越接近硬 max。"""
    a = float(alpha)
    m = np.max(vals, axis=-1, keepdims=True)
    z = np.exp(a * (vals - m))
    return (np.log(np.sum(z, axis=-1, keepdims=True)) / a + m)[..., 0]

def softmin(vals, alpha: float):
    """C¹ 光滑 min。"""
    return -softmax(-vals, alpha)

def sdf_from_implicit(F: np.ndarray, gradF: np.ndarray, eps_norm: float = 1e-9):
    """
    输入:
      F:     (...,)   隐式函数值（外正内负或相反，符号一致即可）
      gradF: (...,2)  对应梯度
    返回:
      s:     (...,)   近似 signed distance: F / ||gradF||
      n:     (...,2)  单位外法线: gradF / ||gradF||
    说明:
      s ≈ F / ||∇F|| 是经典的一阶有理近似，C¹ 连续，LBFGS 友好。
      外/内的符号由 F 的符号决定；力方向使用 n。
    """
    gnorm = np.linalg.norm(gradF, axis=-1, keepdims=True)
    n = gradF / (gnorm + eps_norm)
    s = F / (gnorm[..., 0] + eps_norm)
    return s, n

def rect_implicit_smooth_world(p, c, e, R=None, alpha: float = 20.0, eps: float = 1e-9):
    """
    用四条边的 signed 半空间距离做 smooth-max / smooth-min 的一致性拼接，
    构造 C¹ 连续的“矩形隐式函数”与其梯度，再转为 s, n。
    p:  (...,2) world
    c:  (2,)    中心
    e:  (2,)    半宽半高 (ex, ey)
    R:  (2,2)   world<-local 旋转，None 视为轴对齐
    alpha:      光滑度，20~40 之间通常足够平滑且形状贴合
    返回:
      F, gradF (world 坐标系) —— 供 sdf_from_implicit 使用
    """
    if R is not None:
        p_loc = (p - c) @ R.T
    else:
        p_loc = p - c

    x = p_loc[..., 0]
    y = p_loc[..., 1]
    ex = float(e[0])
    ey = float(e[1])

    d_right = x - ex
    d_left = -x - ex
    d_top = y - ey
    d_bot = -y - ey

    D_out = np.stack([d_right, d_left, d_top, d_bot], axis=-1)
    F_out = softmax(D_out, alpha=alpha)

    D_in = -D_out
    F_in = -softmin(D_in, alpha=alpha)

    inside = (np.abs(x) <= ex + eps) & (np.abs(y) <= ey + eps)
    F = np.where(inside, F_in, F_out)

    def softmax_grad(weights, comps):
        w = weights[..., None]
        return np.sum(w * comps, axis=-2)

    def softmax_weights(vals, alpha):
        a = float(alpha)
        m = np.max(vals, axis=-1, keepdims=True)
        z = np.exp(a * (vals - m))
        w = z / (np.sum(z, axis=-1, keepdims=True) + EPS)
        return w

    W_out = softmax_weights(D_out, alpha)
    W_in = softmax_weights(D_in, alpha)

    comps = np.stack(
        [
            np.stack([np.ones_like(x), np.zeros_like(y)], axis=-1),
            np.stack([-np.ones_like(x), np.zeros_like(y)], axis=-1),
            np.stack([np.zeros_like(x), np.ones_like(y)], axis=-1),
            np.stack([np.zeros_like(x), -np.ones_like(y)], axis=-1),
        ],
        axis=-2,
    )

    grad_out = softmax_grad(W_out, comps)
    grad_in = -softmax_grad(W_in, comps)
    grad_loc = np.where(inside[..., None], grad_in, grad_out)

    if R is not None:
        grad_world = grad_loc @ R
    else:
        grad_world = grad_loc

    return F, grad_world
