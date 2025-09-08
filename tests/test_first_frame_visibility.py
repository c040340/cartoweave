import numpy as np

from cartoweave.compute.solve import solve
from cartoweave.contracts import Anchor, Label, Scene, SolvePack, ActionRecord
from cartoweave.data.api import _derive_active0_from_actions


def _make_pack():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        Label(id=0, kind="point", anchor=Anchor(target="free", mode="xy", xy=(0.0, 0.0))),
        Label(id=1, kind="point", anchor=Anchor(target="free", mode="xy", xy=(0.0, 0.0))),
    ]
    P0 = [(0.0, 0.0), (1.0, 1.0)]
    actions = [
        ActionRecord(id=0, t=0.1, type="mutate"),
        ActionRecord(id=1, t=0.9, type="appear"),
    ]
    active0 = _derive_active0_from_actions(2, actions)
    compute_cfg = {
        "solver": {
            "public": {"use_warmup": True},
            "tuning": {
                "stopping": {"gtol": -1, "ftol": -1, "xtol": -1},
                "warmup": {"steps": 1},
                "lbfgsb": {"maxiter": 1},
            },
        },
        "passes": {"capture": {"every": 1, "mode": "both", "final_always": True}},
    }
    pack = SolvePack(
        L=2,
        P0=P0,
        labels0=labels,
        active0=active0.tolist(),
        scene0=scene,
        cfg={"compute": compute_cfg},
        actions=actions,
        action_num=len(actions),
        behaviors=[],
    )
    return pack


def test_first_frame_visibility():
    pack = _make_pack()
    vp = solve(pack)
    frame0 = vp.frames[0]
    assert np.isfinite(frame0.P[0]).all()
    assert np.isnan(frame0.P[1]).all()
    for arr in frame0.comps.values():
        assert np.all(arr[1] == 0)
