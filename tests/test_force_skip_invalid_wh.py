import numpy as np
from cartoweave.compute.forces import boundary, area_cross, area_softout, area_embed


def test_boundary_wall_skips_zero_width_or_height():
    scene = {"labels": [{"WH": [0.0, 10.0]}], "frame_size": (100.0, 100.0)}
    P = np.array([[50.0, 50.0]])
    E, F, _ = boundary.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)

    scene = {"labels": [{"WH": [10.0, 0.0]}], "frame_size": (100.0, 100.0)}
    P = np.array([[50.0, 50.0]])
    E, F, _ = boundary.evaluate(scene, P, {}, {})
    assert E == 0.0
    assert np.allclose(F, 0.0)


def test_area_cross_skips_zero_width_or_height():
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for wh in ([0.0, 1.0], [1.0, 0.0]):
        scene = {"areas": [square], "labels": [{"WH": wh, "anchor": {"kind": "area", "index": 0}}]}
        P = np.array([[0.8, 0.0]])
        E, F, _ = area_cross.evaluate(scene, P, {}, {})
        assert E == 0.0
        assert np.allclose(F, 0.0)


def test_area_softout_skips_zero_width_or_height():
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for wh in ([0.0, 1.0], [1.0, 0.0]):
        scene = {"areas": [square], "labels": [{"WH": wh, "anchor": {"kind": "area", "index": 0}}]}
        P = np.array([[0.6, 0.0]])
        E, F, _ = area_softout.evaluate(scene, P, {}, {})
        assert E == 0.0
        assert np.allclose(F, 0.0)


def test_area_embed_skips_zero_width_or_height():
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for wh in ([0.0, 1.0], [1.0, 0.0]):
        scene = {"areas": [square], "labels": [{"WH": wh, "anchor": {"kind": "area", "index": 0}}]}
        P = np.array([[0.0, 0.0]])
        E, F, _ = area_embed.evaluate(scene, P, {}, {})
        assert E == 0.0
        assert np.allclose(F, 0.0)
