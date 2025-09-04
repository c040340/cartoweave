# src/cartoweave/engine/forces/term_area_embed.py
from __future__ import annotations
import numpy as np
from . import register

from cartoweave.utils.kernels import (
    EPS_DIST, EPS_NORM, EPS_ABS,
)

from cartoweave.utils.geometry import (
    project_point_to_segment, poly_signed_area, rect_half_extent_along_dir
)
from cartoweave.utils.shape import as_nx2

from cartoweave.utils.numerics import (
    sigmoid_np, d_sigmoid_np, softabs_np, softmin_weights_np,
)

@register("area.embed")
def term_area_embed(scene, P: np.ndarray, cfg, phase="pre_anchor"):
    """
    严格配对版本（FD 友好）：
    - 对每条边 k，建立垂向“目标深度”弹簧：E_perp_k = 0.5 * k_embed * (s_k - s*_k)^2
      其中 s_k = n_k · (C - q_k)，s*_k = (2*ratio_in - 1) * r_n_k，r_n_k 取矩形沿 n_k 的半径。
    - 对每条边的切向加一个 gated 二次项：E_tan_k = 0.5 * k_tan * g_k * u_k^2
      其中 u_k = t_k · (C - q_k)，g_k = sigmoid(((r_n_k + slack) - softabs(s_k))/eta)。
    - 最后按 softmin 权重 w_k 聚合：E = sum_k w_k * (E_perp_k + E_tan_k)
      权重基于 v_k = softabs(s_k) 的 softmax：w_k = exp(-β v_k) / Σ exp(-β v_j)。
      这样 ∂w/∂v = (-β)*(diag(w) - w w^T)，可写出链式导数。
    备注：
      - 投影点 q_k 对 C 的导数在 s_k 的垂向上为 0（n·∂q/∂C = 0，内点时成立），
        这里采用与 ln.rect 同样的近似：ds_k/dC ≈ n_k，du_k/dC ≈ t_k，数值已验证 FD 可过。
    """
    if phase != "pre_anchor" or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {}

    labels = scene.get("labels", [])
    areas  = scene.get("areas", [])
    N = P.shape[0]
    WH_raw = scene.get("WH")
    if WH_raw is None:
        WH_raw = np.zeros((N, 2))
    WH = as_nx2(WH_raw, N, "WH")

    k_embed   = float(cfg.get("area.k.embed", 200.0))
    k_tan     = float(cfg.get("area.k.tan",   30.0))
    ratio_in  = float(cfg.get("area.embed.ratio_in", 0.60))
    gate_eta  = float(cfg.get("area.tan.gate.eta", 2.0))
    gate_slack= float(cfg.get("area.tan.gate.slack", 1.0))
    beta_edge = float(cfg.get("area.embed.beta_edge", 6.0))  # softmin 温度（稍大些更稳）
    eps_abs   = float(cfg.get("eps.abs", EPS_ABS))
    eps_div   = 1e-9

    F = np.zeros_like(P, float)
    E_total = 0.0
    S = [[] for _ in range(P.shape[0])]

    for i, lab in enumerate(labels):
        if lab.get("anchor_kind") != "area":
            continue
        ai = int(lab.get("anchor_index", -1))
        if ai < 0 or ai >= len(areas):
            continue
        poly = areas[ai].get("polygon", None)
        if poly is None:
            continue
        arr = np.asarray(poly, float).reshape(-1, 2)
        nE = arr.shape[0]
        if nE < 3:
            continue

        # label 尺寸/中心
        w, h = float(WH[i,0]), float(WH[i,1])
        if w <= 0.0 and h <= 0.0:
            continue
        hx, hy = 0.5*w, 0.5*h
        cx, cy = float(P[i,0]), float(P[i,1])

        # 多边形朝向（决定内法向）
        ccw = (poly_signed_area(arr) > 0.0)

        # 逐边收集量
        s_list  = np.zeros(nE)   # normal signed distance of label center to edge
        u_list  = np.zeros(nE)   # tangential distance
        rn_list = np.zeros(nE)   # rect half extent along each edge's inward normal
        n_list  = np.zeros((nE, 2))  # inward normal
        t_list  = np.zeros((nE, 2))  # unit tangent
        q_list  = np.zeros((nE, 2))  # closest points
        tparam  = np.zeros(nE)

        for k in range(nE):
            ax, ay = float(arr[k,0]), float(arr[k,1])
            bx, by = float(arr[(k+1)%nE,0]), float(arr[(k+1)%nE,1])
            qx, qy, t, tx, ty = project_point_to_segment(cx, cy, ax, ay, bx, by)
            nx_in, ny_in = (-ty, tx) if ccw else (ty, -tx)

            dx, dy = (cx - qx), (cy - qy)
            s_k = nx_in*dx + ny_in*dy
            u_k = tx*dx + ty*dy

            r_n_k = rect_half_extent_along_dir(w, h, nx_in, ny_in, eps_abs)

            s_list[k]  = s_k
            u_list[k]  = u_k
            rn_list[k] = r_n_k
            n_list[k]  = (nx_in, ny_in)
            t_list[k]  = (tx, ty)
            q_list[k]  = (qx, qy)
            tparam[k]  = t

        # softmin 权重：w = softmax(-β * softabs(s))
        sabs = softabs_np(s_list, eps_abs)
        wgt = softmin_weights_np(sabs, beta_edge)
        # dw/dv = (-β) * (diag(w) - w w^T)
        J = (-beta_edge) * (np.diag(wgt) - np.outer(wgt, wgt))  # (nE,nE)
        # dv/ds = s / softabs(s)
        dv_ds = s_list / np.maximum(sabs, eps_div)              # (nE,)

        # 每条边的能量（配对）
        s_star = (2.0*ratio_in - 1.0) * rn_list                # (nE,)
        ds     = s_list - s_star                               # (nE,)
        E_perp = 0.5 * k_embed * (ds*ds)                       # (nE,)

        # 切向 gated
        inv_eta = 1.0 / max(gate_eta, eps_div)
        x = ((rn_list + gate_slack) - sabs) * inv_eta
        g = sigmoid_np(x)
        if not (np.isfinite(g).all() and np.isfinite(wgt).all()):
            raise FloatingPointError("area_embed: non-finite gate/weights")
        E_tan = 0.5 * k_tan * g * (u_list*u_list)              # (nE,)

        # 总能量：按权重加权
        E_k = E_perp + E_tan                                   # (nE,)
        E = float((wgt * E_k).sum())
        E_total += E

        # 梯度：∂E/∂C = Σ_k [ w_k * ∂E_k/∂C + E_k * ∂w_k/∂C ]
        # 先算各项的 ∂E_k/∂C
        # ∂E_perp_k/∂C = k_embed * ds_k * ∂s_k/∂C， 取 ∂s_k/∂C ≈ n_k
        dEperp_dC = (k_embed * ds)[:, None] * n_list           # (nE,2)

        # ∂E_tan_k/∂C = 0.5*k_tan*(u_k^2)*∂g/∂C + k_tan*g*u_k*∂u_k/∂C
        # g = σ(((r_n+slack) - softabs(s))/η)
        gprime = d_sigmoid_np(x) * (-inv_eta) * dv_ds          # (nE,)
        dEtan_dC = (0.5 * k_tan * (u_list*u_list) * gprime)[:,None] * n_list \
                 + (k_tan * g * u_list)[:,None] * t_list       # (nE,2)

        dEk_dC = dEperp_dC + dEtan_dC                          # (nE,2)

        # 再算 ∂w/∂C：dw/dC = J · dv/dC = J · (dv/ds * ds/dC)
        # 其中 ds/dC ≈ n_k
        dv_dC = (dv_ds[:,None]) * n_list                       # (nE,2)
        # 按分量把 J 作用到 dv/dC： (nE,nE) · (nE,2) -> (nE,2)
        dw_dC = J @ dv_dC

        # 汇总 ∂E/∂C
        # Σ_k w_k * dEk/dC  +  Σ_k E_k * dw_k/dC
        grad = (wgt[:,None] * dEk_dC).sum(axis=0) + (E_k[:,None] * dw_dC).sum(axis=0)

        # 力 = -∇E
        F[i,0] += -grad[0]
        F[i,1] += -grad[1]

        # 记录（任选最“近”边）
        k_min = int(np.argmin(sabs))
        S[i].append((
            int(ai), float(-grad[0]), float(-grad[1]),
            float(E), k_min, float(s_list[k_min]), float(u_list[k_min]),
            float(wgt[k_min])
        ))

    return float(E_total), F, {"area_embed": S}
