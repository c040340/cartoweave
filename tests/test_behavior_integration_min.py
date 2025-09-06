import numpy as np
from cartoweave.contracts.solvepack import SolvePack
from cartoweave.compute.passes.behavior_pass import BehaviorPass
from cartoweave.compute.solver.solve_layer import SolveContext, run_iters
from cartoweave.compute.geom.anchor import anchor_xy


def fake_energy(P, labels, scene, active, cfg):
    g = P.copy()
    E = 0.5 * float((P * P).sum())
    return E, g, {}


def _make_pack(anchor_xy_point=(5.0, 5.0)):
    scene = {
        "frame_size": (10, 10),
        "labels": [{"anchor_kind": "point", "anchor_index": 0}],
        "WH": np.array([[1.0, 1.0]]),
        "points": np.array([anchor_xy_point]),
        "lines": [],
        "areas": [],
    }
    cfg = {"behaviors": []}
    return SolvePack(1, np.zeros((1, 2)), np.array([False]), scene=scene, cfg=cfg)


def _run(pack, behaviors, cfg):
    pack.cfg["behaviors"] = behaviors
    bp = BehaviorPass(pack)
    P_prev = pack.P0.copy()
    results = []
    for k, beh in enumerate(behaviors):
        P0, labels_k, active_k, scene_k = bp.begin_behavior(k, beh, P_prev, cfg)
        ctx = SolveContext(labels=labels_k, scene=scene_k, active=active_k, cfg=cfg,
                           iters=beh.get("iters", 0), params=beh.get("params", {}))
        P_star, _ = run_iters(P0, ctx, fake_energy, report=False)
        results.append((P0.copy(), P_star.copy()))
        bp.end_behavior(k, P_star)
        P_prev = P_star
    return results


def test_first_activation_snaps_once():
    pack = _make_pack()
    cfg = {"behavior": {"place_on_first_activation": True}}
    behaviors = [
        {"iters": 1, "ops": {"activate": [0]}, "params": {"step": 0.1}},
        {"iters": 1, "ops": {}, "params": {"step": 0.1}},
    ]
    traj = _run(pack, behaviors, cfg)
    anchor = anchor_xy(pack.scene["labels"][0], pack.scene)
    assert np.allclose(traj[0][0][0], anchor, atol=1e-6)
    assert np.allclose(traj[1][0][0], traj[0][1][0], atol=1e-6)


def test_size_change_keeps_position():
    pack = _make_pack()
    cfg = {"behavior": {"place_on_first_activation": True}}
    behaviors = [
        {"iters": 1, "ops": {"activate": [0]}, "params": {"step": 0.1}},
        {"iters": 1, "ops": {"mutate": [{"id": 0, "set": {"WH": [2.0, 2.0]}}]}, "params": {"step": 0.1}},
    ]
    traj = _run(pack, behaviors, cfg)
    assert np.allclose(traj[1][0][0], traj[0][1][0], atol=1e-6)
