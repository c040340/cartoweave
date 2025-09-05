import numpy as np
from cartoweave.compute import SolvePack, solve
from cartoweave.compute.eval import energy_and_grad_full


def test_weights_scaling(P0, mask, scene, L):
    rng = np.random.default_rng(0)
    P = rng.standard_normal((L, 2)).astype(float)

    # 只开 pre_anchor：ll.disk
    cfg0 = {"compute": {"weights": {"ll.disk": 1.0}, "eps": {"numeric": 1e-12}}}
    cfgw = {"compute": {"weights": {"ll.disk": 0.5}, "eps": {"numeric": 1e-12}}}

    E0, G0, C0, _ = energy_and_grad_full(P, scene, mask, cfg0)
    Ew, Gw, Cw, _ = energy_and_grad_full(P, scene, mask, cfgw)

    # 权重缩放
    r = np.linalg.norm(Cw["ll.disk"]) / (np.linalg.norm(C0["ll.disk"]) + 1e-12)
    assert np.isclose(r, 0.5, rtol=1e-5, atol=1e-8)
    # 一致性：G 仍等于 -ΣF
    assert np.allclose(Gw, -sum(Cw.values()), atol=1e-6)

def test_nan_guard_and_grad_clip_applied(P0, mask, scene, L, monkeypatch):
    # 用一个会返回 NaN/大梯度的能量函数，验证 pass 修复与裁剪
    def bad_energy(P, scene, mask, cfg):
        import numpy as np
        G = np.full_like(P, np.inf)
        comps = {"foo": -G.copy()}
        return float("nan"), G, comps, {}
    from cartoweave.compute import SolvePack, solve
    cfg = {"compute": {"passes": {"grad_clip": {"max_norm": 1.0}, "capture": {"every": 1}}}}
    sp = SolvePack(L=L, P0=P0, active_mask0=mask, scene=scene, params={"max_iter":1},
                   energy_and_grad=bad_energy, cfg=cfg)
    sp.passes=["schedule", "grad_clip", "nan_guard", "capture"]
    vp = solve(sp)
    assert "pass_stats" in vp.summary
    ps = vp.summary["pass_stats"]
    assert ps.get("NaNGuardPass",{}).get("fixed_frames",0) >= 1
    assert ps.get("GradClipPass",{}).get("clipped_frames",0) >= 1
    # 且保持 G ≈ -Σcomps
    import numpy as np
    Gsum = -sum(vp.last.comps.values())
    assert np.allclose(vp.last.G, Gsum, atol=1e-6)
