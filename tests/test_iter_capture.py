from cartoweave.compute.solve import solve
from cartoweave.data.api import make_solvepack_from_data_defaults


def _make_pack(iters: int, every: int = 1, limit: int | None = None, final: bool = True):
    compute_cfg = {
        "solver": {
            "tuning": {
                "stopping": {"gtol": -1, "ftol": -1, "xtol": -1},
                "warmup": {"steps": iters},
            }
        },
        "passes": {
            "capture": {
                "every": every,
                "limit": limit,
                "final_always": final,
                "mode": "warmup",
            }
        },
    }
    pack = make_solvepack_from_data_defaults(compute_cfg=compute_cfg)
    pack.actions = pack.actions[:1]
    pack.action_num = 1
    return pack


def test_iter_frames_increase():
    pack = _make_pack(7, every=1, final=True)
    vp = solve(pack)
    assert vp.num_frames() >= 7
    assert vp.passes[0].t_start == 0
    assert vp.passes[0].t_end == vp.num_frames()


def test_cadence_and_final_toggle():
    pack = _make_pack(5, every=3, final=True)
    vp = solve(pack)
    fins = [fr.meta["frame_in_pass"] for fr in vp.frames]
    assert fins == [0, 3, "final"]

    pack2 = _make_pack(5, every=3, final=False)
    vp2 = solve(pack2)
    fins2 = [fr.meta["frame_in_pass"] for fr in vp2.frames]
    assert fins2 == [0, 3]


def test_limit_enforced():
    pack = _make_pack(50, every=1, limit=5, final=True)
    vp = solve(pack)
    assert vp.num_frames() <= 5


def test_no_final_duplication():
    pack = _make_pack(10, every=1, final=True)
    vp = solve(pack)
    fins = [fr.meta["frame_in_pass"] for fr in vp.frames]
    assert fins[-1] == 9
    assert "final" not in fins
