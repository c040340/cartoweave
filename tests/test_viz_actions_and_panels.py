import matplotlib.pyplot as plt
from cartoweave.viz.backend import use_compatible_backend
from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve
from cartoweave.viz.build_viz_payload import build_viz_payload
from cartoweave.viz.panels import draw_layout
from cartoweave.viz.defaults import VIZ_DEFAULTS


use_compatible_backend()


def _make_viewpack(seed: int, wrap):
    cfg = {
        "data": {
            "source": "generate",
            "generate": {
                "num_points": 2,
                "num_lines": 0,
                "num_areas": 0,
                "num_steps": 1,
            },
        },
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }
    sp = build_solvepack_from_config(cfg, seed=seed)
    sp.scene0.WH[:] = 10.0
    vp = solve(wrap(sp))
    return sp, vp


def test_viz_actions_smoke(as_legacy_pack):
    sp, vp = _make_viewpack(0, as_legacy_pack)
    records = [
        {"P": fr.P, "comps": fr.comps, "meta": {"eval_index": fr.i}}
        for fr in vp.frames
    ]
    info = {
        "history": {
            "records": records,
            "scene_steps": [
                {"name": "s0", "rec_start": 0, "rec_end": len(records)}
            ],
        },
        "scene": sp.scene0.to_dict(),
    }
    payload = build_viz_payload(info)
    assert isinstance(payload, dict)
    assert "frames" in payload and len(payload["frames"]) >= 1
    assert "actions" in payload and isinstance(payload["actions"], list)


def test_viz_panels_smoke(as_legacy_pack):
    sp, vp = _make_viewpack(1, as_legacy_pack)
    fig, ax = plt.subplots()
    try:
        patches = draw_layout(
            ax,
            vp.last.P,
            sp.scene0.labels,
            sp.scene0.WH,
            frame_w=sp.scene0.frame_size[0],
            frame_h=sp.scene0.frame_size[1],
            viz_layout=VIZ_DEFAULTS["layout"],
        )
    finally:
        plt.close(fig)
    assert isinstance(patches, list)
    assert len(patches) >= 1

