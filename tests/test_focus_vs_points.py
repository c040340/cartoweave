import numpy as np
from cartoweave.orchestrators.solve_plan import run_solve_plan

def base_cfg():
    return {
        "ll.geom": "rect", "ll.k.repulse": 150.0, "ll.k.inside": 50.0,
        "boundary.k.wall": 0.0,
        "anchor.k.spring": 0.0,
        # 椭圆 Huber focus
        "focus.k.attract": 40.0,
        "focus.center": np.array([400.,300.]),
        "focus.sigma.x": 80.0, "focus.sigma.y": 120.0,
        "focus.delta": 10.0,
    }

def make_scene(points):
    return dict(
        frame=0, frame_size=(800, 600),
        labels_init=np.array([[300.,300.]], float),
        WH=np.array([[60.,24.]], float),
        labels=[{"anchor_kind":"none"}],
        points=points,
        lines=np.zeros((0,2,2), float), areas=[],
        anchors=np.zeros((1,2), float),
    )

def test_focus_does_not_depend_on_points_when_pl_zero():
    cfg = base_cfg()
    cfg["pl.k.repulse"] = 0.0
    cfg["pl.k.inside"]  = 0.0

    scene_with_pts = make_scene(points=np.array([[350.,300.],[500.,500.]], float))
    scene_no_pts   = make_scene(points=np.zeros((0,2), float))

    P1, _ = run_solve_plan(scene_with_pts, cfg, [{"name":"main"}], mode="hybrid")
    P2, _ = run_solve_plan(scene_no_pts,   cfg, [{"name":"main"}], mode="hybrid")

    assert np.allclose(P1, P2), f"Focus must be independent of points when pl.*=0"

def test_points_affect_when_pl_enabled():
    cfg = base_cfg()
    cfg["pl.k.repulse"] = 300.0   # 开启点↔label 排斥
    cfg["pl.k.inside"]  = 50.0

    scene_with_pts = make_scene(points=np.array([[330.,300.]], float))   # 点靠得近
    scene_no_pts   = make_scene(points=np.zeros((0,2), float))

    P1, _ = run_solve_plan(scene_with_pts, cfg, [{"name":"main"}], mode="hybrid")
    P2, _ = run_solve_plan(scene_no_pts,   cfg, [{"name":"main"}], mode="hybrid")

    diff = float(np.linalg.norm(P1 - P2))
    assert diff > 1e-3, "When pl.* > 0, points should change the solution"
