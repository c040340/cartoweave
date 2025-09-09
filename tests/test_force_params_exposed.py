import copy
import numpy as np
from cartoweave.config.loader import load_compute_config
from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.contracts.solvepack import Anchor, Label, Scene


def test_pl_rect_exponent_affects_signal():
    base_cfg = load_compute_config()

    scene = Scene(frame_size=(100.0, 100.0), points=[(0.0, 0.0)])
    labels = [
        Label(
            id=0,
            kind="point",
            WH=(10.0, 10.0),
            anchor=Anchor(target="point", index=0, mode="exact"),
        )
    ]
    P = np.array([[5.0, 5.0]], float)
    active = np.array([True])

    def eval_energy(exp: float) -> float:
        cfg = copy.deepcopy(base_cfg)
        cfg["compute"]["public"]["forces"]["pl"]["rect"]["kernel"]["exponent"] = exp
        E, _, _ = energy_and_grad_full(P, labels, scene, active, cfg["compute"])
        return E

    e1 = eval_energy(1.5)
    e2 = eval_energy(3.0)
    assert e1 != e2


def test_area_cross_min_gap_changes_contact():
    base_cfg = load_compute_config()

    area0 = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    area1 = [(11.0, 0.0), (21.0, 0.0), (21.0, 10.0), (11.0, 10.0)]
    scene = Scene(frame_size=(40.0, 20.0), areas=[area0, area1])
    labels = [
        Label(
            id=0,
            kind="area",
            WH=(8.0, 8.0),
            anchor=Anchor(target="area", index=0, mode="centroid"),
        ),
        Label(
            id=1,
            kind="area",
            WH=(8.0, 8.0),
            anchor=Anchor(target="area", index=1, mode="centroid"),
        ),
    ]
    P = np.array([[5.0, 5.0], [16.0, 5.0]], float)
    active = np.array([True, True])

    def eval_energy(min_gap: float) -> float:
        cfg = copy.deepcopy(base_cfg)
        cfg["compute"]["public"]["forces"]["area"]["cross"]["min_gap"] = min_gap
        E, _, _ = energy_and_grad_full(P, labels, scene, active, cfg["compute"])
        return E

    e1 = eval_energy(0.0)
    e2 = eval_energy(3.0)
    assert e1 != e2

