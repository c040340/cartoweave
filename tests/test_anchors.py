import numpy as np
from cartoweave.labels import anchor_xy, init_position
from cartoweave.layout_utils.geometry import (
    poly_centroid,
    project_point_to_segment,
    area_anchor_from_centroid_nearest_edge,
)

FS = (800.0, 600.0)


def test_anchor_point_trivial():
    pt = np.array([[100.0, 50.0]], float)
    qx, qy = anchor_xy("point", 0, {"points": pt}, FS)
    assert qx == 100.0 and qy == 50.0


def test_anchor_line_lshape():
    poly = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]], float)
    qx, qy = anchor_xy("line", 0, {"lines": [poly]}, FS)
    assert qx == 100.0 and np.isclose(qy, 25.49019607843137)


def test_anchor_line_folded_plus_tail():
    poly = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 200.0]],
        float,
    )
    qx, qy = anchor_xy("line", 0, {"lines": [poly]}, FS)
    assert np.isclose(qx, 49.504950495049506) and qy == 100.0


def test_anchor_area_centroid_projection():
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], float)
    qx, qy, meta = anchor_xy("area", 0, {"areas": [square]}, FS, with_meta=True)
    assert qx == 50.0 and qy == 0.0
    assert meta["seg_index"] == 0
    assert np.allclose(meta["normal_in"], (0.0, 1.0))


def test_area_winding_normals():
    ccw = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], float)
    cw = ccw[::-1]
    _, _, meta1 = anchor_xy("area", 0, {"areas": [ccw]}, FS, with_meta=True)
    _, _, meta2 = anchor_xy("area", 0, {"areas": [cw]}, FS, with_meta=True)
    assert np.allclose(meta1["normal_in"], (0.0, 1.0))
    assert np.allclose(meta2["normal_in"], (0.0, -1.0))


def test_project_point_zero_length_segment():
    qx, qy, t = project_point_to_segment(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert qx == 0.0 and qy == 0.0 and t == 0.0


def test_poly_centroid_degenerate():
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], float)
    cx, cy = poly_centroid(poly)
    assert np.isfinite(cx) and np.isfinite(cy)


def test_area_anchor_concave():
    poly = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 20.0],
            [20.0, 20.0],
            [20.0, 80.0],
            [100.0, 80.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ],
        float,
    )
    info = area_anchor_from_centroid_nearest_edge(poly)
    assert info["seg_index"] == 3
    assert info["qx"] == 20.0 and info["qy"] == 50.0


def test_init_behavior_new_vs_existing():
    anchor = (100.0, 100.0)
    prev = (120.0, 130.0)
    out_prev = init_position("point", anchor, FS, prev=prev)
    np.testing.assert_allclose(out_prev, prev)
    out_new = init_position("point", anchor, FS)
    assert not np.allclose(out_new, anchor)
    d = np.hypot(*(out_new - np.array(anchor)))
    assert 1.0 <= d <= 5.0
    out_circle = init_position("point", anchor, FS, mode="circle")
    np.testing.assert_allclose(out_circle, anchor)
