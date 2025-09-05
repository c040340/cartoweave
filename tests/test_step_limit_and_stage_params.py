from cartoweave.compute import SolvePack, solve
from cartoweave.compute.eval import energy_and_grad_full
import numpy as np

def test_step_limit_injected_and_recorded(monkeypatch, P0, mask, scene, L):
    captured_params = []
    def fake_solver(mode, *, P0, scene, active_mask, params, energy_and_grad, recorder):
        # 伪造 2 次 evaluation，且回调一帧
        captured_params.append(params.copy())
        E,G,comps,_ = energy_and_grad(P0, scene, active_mask, params)
        recorder({"P":P0, "E":E, "G":G, "comps":comps, "mask":active_mask, "ls_iters":1, "alpha":0.5})
        return {"P_final":P0, "E_last":E, "converged":True, "stop_reason":"test", "iters":1}
    import cartoweave.compute.optim as optim
    monkeypatch.setattr(optim, "run_via_legacy_solver", fake_solver)

    params={"max_iter":2, "terms":{"anchor":{"spring":{"k":5.0}}}}
    sp = SolvePack(L=L, P0=P0, active_mask0=mask, scene=scene, params=params,
                   energy_and_grad=energy_and_grad_full)
    sp.schedule=[{"solver":"lbfgs","iters":2}]
    sp.passes=["schedule", {"name":"step_limit","args":{"max_step":2.5,"damping":0.8}},
               {"name":"capture","args":{"every":1}}]
    vp = solve(sp)
    # 验注入
    assert captured_params and "step_limit" in captured_params[0] and "damping" in captured_params[0]
    # 验 summary
    ps = vp.summary.get("pass_stats", {}).get("StepLimitPass", {})
    assert ps.get("applied_stages", 0) >= 1
    assert ps.get("max_step") == 2.5
    assert ps.get("damping") == 0.8
