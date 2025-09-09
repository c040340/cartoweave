import numpy as np
from cartoweave.compute.forces import point_label, line_label, area_cross, area_softout


def test_point_without_active_label_exerts_no_force():
    scene = {"points": [(0.0, 0.0)], "labels": [{"WH": [1.0, 1.0]}]}
    P = np.array([[0.5, 0.0]])
    E, F, _ = point_label.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)


def test_line_without_active_label_exerts_no_force():
    scene = {
        "lines": [np.array([[-1.0, 0.0], [1.0, 0.0]])],
        "labels": [{"WH": [1.0, 1.0]}],
    }
    P = np.array([[0.0, 0.5]])
    E, F, _ = line_label.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)


def test_area_cross_without_active_label_exerts_no_force():
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    scene = {"areas": [square], "labels": [{"WH": [1.0, 1.0]}]}
    P = np.array([[0.8, 0.0]])
    E, F, _ = area_cross.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)


def test_area_softout_without_active_label_exerts_no_force():
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    scene = {"areas": [square], "labels": [{"WH": [1.0, 1.0]}]}
    P = np.array([[0.6, 0.0]])
    E, F, _ = area_softout.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)
