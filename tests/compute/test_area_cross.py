import copy
import numpy as np
from cartoweave.compute.forces import area_cross
from cartoweave.config.loader import load_compute_config
from cartoweave.contracts.solvepack import Scene, Label, Anchor

def _base_cfg():
    return load_compute_config()["compute"]

def test_degenerate_edge_far_label():
    cfg = _base_cfg()
    # area0 has a degenerate edge (first two points identical)
    area0 = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    area1 = [(1000.0, 1000.0), (1010.0, 1000.0), (1010.0, 1010.0), (1000.0, 1010.0)]
    scene = Scene(frame_size=(2000.0, 2000.0), areas=[area0, area1])
    labels = [
        Label(id=0, kind="area", WH=(8.0, 8.0), anchor=Anchor(target="area", index=0, mode="centroid")),
        Label(id=1, kind="area", WH=(8.0, 8.0), anchor=Anchor(target="area", index=1, mode="centroid")),
    ]
    P = np.array([[5.0, 5.0], [1005.0, 1005.0]], float)
    sc = scene.model_dump()
    sc["labels"] = labels
    E, F, info = area_cross.evaluate(sc, P, {}, cfg)
    stats = info["stats"]
    assert np.all(np.isfinite(F))
    assert stats["max_force_norm"] < 1e-3
    assert stats["skipped_short_edges"] > 0

def test_aabb_preclip_rejects_far_edges():
    cfg = _base_cfg()
    area0 = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    area1 = [(1000.0, 1000.0), (1010.0, 1000.0), (1010.0, 1010.0), (1000.0, 1010.0)]
    scene = Scene(frame_size=(2000.0, 2000.0), areas=[area0, area1])
    labels = [
        Label(id=0, kind="area", WH=(8.0, 8.0), anchor=Anchor(target="area", index=0, mode="centroid")),
        Label(id=1, kind="area", WH=(8.0, 8.0), anchor=Anchor(target="area", index=1, mode="centroid")),
    ]
    P = np.array([[5.0, 5.0], [1005.0, 1005.0]], float)
    sc = scene.model_dump()
    sc["labels"] = labels
    E, F, info = area_cross.evaluate(sc, P, {}, cfg)
    stats = info["stats"]
    assert stats["aabb_rejected_edges"] > 0
    assert stats["max_force_norm"] < 1e-6

def test_parameter_trend_eta_beta():
    cfg = _base_cfg()
    # Place the second area far away so forces are near zero
    area0 = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    area1 = [(100.0, 0.0), (110.0, 0.0), (110.0, 10.0), (100.0, 10.0)]
    scene = Scene(frame_size=(200.0, 20.0), areas=[area0, area1])
    labels = [
        Label(id=0, kind="area", WH=(8.0, 8.0), anchor=Anchor(target="area", index=0, mode="centroid")),
        Label(id=1, kind="area", WH=(8.0, 8.0), anchor=Anchor(target="area", index=1, mode="centroid")),
    ]
    P = np.array([[5.0, 5.0], [105.0, 5.0]], float)
    sc = scene.model_dump()
    sc["labels"] = labels
    _, _, info0 = area_cross.evaluate(sc, P, {}, cfg)
    base = info0["stats"]["max_force_norm"]
    cfg_eta = copy.deepcopy(cfg)
    cfg_eta["public"]["forces"]["area"]["cross"]["eta"] = 8.0
    _, _, info_eta = area_cross.evaluate(sc, P, {}, cfg_eta)
    cfg_beta = copy.deepcopy(cfg)
    cfg_beta["public"]["forces"]["area"]["cross"]["beta_smax"] = 2.0
    _, _, info_beta = area_cross.evaluate(sc, P, {}, cfg_beta)
    assert info_eta["stats"]["max_force_norm"] <= base + 1e-9
    assert info_beta["stats"]["max_force_norm"] <= base + 1e-9
