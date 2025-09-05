import numpy as np
from cartoweave.data.primitives.polygons import polygon_self_intersect, generate_polygon_by_area
from cartoweave.data.config import AreaGenCfg


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_polygon_generator_area_and_non_intersection():
    rng = np.random.default_rng(0)
    cfg = AreaGenCfg()
    frame = (200, 100)
    S_target = 1000.0
    for _ in range(20):
        poly = generate_polygon_by_area(
            rng,
            frame,
            S_target,
            cfg.inset_margin_scale * (frame[0] ** 2 + frame[1] ** 2) ** 0.5,
            cfg.min_edge_spacing_scale * (frame[0] ** 2 + frame[1] ** 2) ** 0.5,
            (cfg.n_vertices_min, cfg.n_vertices_max),
        )
        area = abs(_poly_area(poly))
        assert not polygon_self_intersect(poly)
        assert 0.2 * S_target < area < 5.0 * S_target
