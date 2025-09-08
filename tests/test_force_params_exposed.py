import copy

import numpy as np
from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.config.loader import load_compute_config
from cartoweave.compute.eval import energy_and_grad_full


def _make_pack(cfg: dict):
    return make_solvepack_from_data_defaults(compute_cfg=cfg["compute"])


def solve_with_override(base_cfg, path, key, value):
    cfg = copy.deepcopy(base_cfg)
    d = cfg["compute"]["public"]["forces"]
    for p in path[:-1]:
        d = d[p]
    if path[-1] not in d:
        d[path[-1]] = {}
    d[path[-1]][key] = value
    pack = _make_pack(cfg)
    P = np.asarray(pack.P0, float)
    labels = pack.labels0
    scene = pack.scene0.model_dump()
    active = np.asarray(pack.active0, bool)
    E, G, _ = energy_and_grad_full(P, labels, scene, active, cfg["compute"])
    return float(np.linalg.norm(G)), E


def test_pl_rect_exponent_affects_signal():
    base_cfg = load_compute_config()
    _, e1 = solve_with_override(base_cfg, ["pl", "rect", "kernel"], "exponent", 1.5)
    _, e2 = solve_with_override(base_cfg, ["pl", "rect", "kernel"], "exponent", 3.0)
    assert e1 != e2


def test_area_cross_min_gap_changes_contact():
    base_cfg = load_compute_config()
    _, e1 = solve_with_override(base_cfg, ["area", "cross"], "min_gap", 0.0)
    _, e2 = solve_with_override(base_cfg, ["area", "cross"], "min_gap", 3.0)
    assert e1 != e2

