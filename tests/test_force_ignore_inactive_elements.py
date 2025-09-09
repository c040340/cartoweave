import numpy as np
from cartoweave.compute.forces import (
    point_label,
    line_label,
    area_cross,
    area_softout,
    area_embed,
    label_label_disk,
)


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


def test_point_label_ignores_points_from_inactive_labels():
    scene = {
        "points": [(0.0, 0.0)],
        "labels": [
            {"WH": [1.0, 1.0], "anchor": {"kind": "point", "index": 0}},
            {"WH": [1.0, 1.0]},
        ],
        "_active_ids_solver": [1],
    }
    P = np.array([[0.0, 0.0], [0.5, 0.0]])
    E, F, _ = point_label.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)


def test_line_label_ignores_lines_from_inactive_labels():
    scene = {
        "lines": [np.array([[-1.0, 0.0], [1.0, 0.0]])],
        "labels": [
            {"WH": [1.0, 1.0], "anchor": {"kind": "line", "index": 0}},
            {"WH": [1.0, 1.0]},
        ],
        "_active_ids_solver": [1],
    }
    P = np.array([[0.0, 0.0], [0.0, 0.5]])
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


def test_area_embed_inactive_label_exerts_no_force():
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    scene = {
        "areas": [square],
        "labels": [
            {"WH": [1.0, 1.0], "anchor": {"kind": "area", "index": 0}},
        ],
        "_active_ids_solver": [],
    }
    P = np.array([[0.0, 0.0]])
    E, F, _ = area_embed.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)


def test_area_cross_ignores_area_without_active_labels():
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    far_square = square + np.array([100.0, 0.0])
    scene = {
        "areas": [far_square, square],
        "labels": [
            {"WH": [1.0, 1.0], "anchor": {"kind": "area", "index": 0}},
            {"WH": [1.0, 1.0], "anchor": {"kind": "area", "index": 1}},
        ],
        "_active_ids_solver": [0],
    }
    P = np.array([[0.8, 0.0], [0.0, 0.0]])
    E, F, _ = area_cross.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)


def test_area_softout_ignores_area_without_active_labels():
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    far_square = square + np.array([100.0, 0.0])
    scene = {
        "areas": [far_square, square],
        "labels": [
            {"WH": [1.0, 1.0], "anchor": {"kind": "area", "index": 0}},
            {"WH": [1.0, 1.0], "anchor": {"kind": "area", "index": 1}},
        ],
        "_active_ids_solver": [0],
    }
    P = np.array([[0.6, 0.0], [0.0, 0.0]])
    E, F, _ = area_softout.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)


def test_ll_disk_without_active_label_exerts_no_force():
    scene = {
        "labels": [{"WH": [10.0, 10.0]}, {"WH": [10.0, 10.0]}],
        "_active_ids_solver": [0],
    }
    P = np.array([[0.0, 0.0], [15.0, 0.0]], float)
    E, F, _ = label_label_disk.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)
