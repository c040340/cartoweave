import numpy as np

from cartoweave.compute.solve import solve
from cartoweave.contracts import Anchor, Label, Scene, SolvePack, ActionRecord
from cartoweave.data.api import _derive_active0_from_actions


def _make_pack():
    scene = Scene(frame_size=(10.0, 10.0))
    labels = [
        Label(
            id=0,
            kind="point",
            anchor=Anchor(target="free", mode="xy", xy=(0.0, 0.0)),
            WH=(100.0, 50.0),
            meta={"mode": "old"},
        ),
    ]
    P0 = [(0.0, 0.0)]
    actions = [
        ActionRecord(id=0, t=0.1, type="appear", WH_to=(100.0, 50.0), kind_to="old"),
        ActionRecord(id=0, t=0.2, type="mutate", WH_to=(80.0, 40.0), kind_to="new"),
    ]
    active0 = _derive_active0_from_actions(1, actions)
    compute_cfg = {
        "solver": {
            "public": {"use_warmup": True},
            "tuning": {
                "stop": {"ftol": -1, "xtol": -1, "max_stall_iters": None},
                "warmup": {"steps": 1},
                "lbfgsb": {"lbfgs_maxiter": 1},
            },
        },
        "passes": {"capture": {"every": 1, "mode": "both", "final_always": True}},
    }
    pack = SolvePack(
        L=1,
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


def test_mutate_updates_viewpack():
    pack = _make_pack()
    vp = solve(pack)
    assert vp.WH is not None
    np.testing.assert_allclose(vp.WH[0], np.array([80.0, 40.0]))
    assert vp.labels[0].get("meta", {}).get("mode") == "new"
    wh0 = vp.frames[0].meta.get("WH")
    wh_last = vp.frames[-1].meta.get("WH")
    assert wh0 is not None and wh_last is not None
    np.testing.assert_allclose(wh0[0], np.array([100.0, 50.0]))
    np.testing.assert_allclose(wh_last[0], np.array([80.0, 40.0]))
