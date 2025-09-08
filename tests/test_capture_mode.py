from cartoweave.compute.solve import solve
from cartoweave.data.api import make_solvepack_from_data_defaults


def _make_pack(mode: str, use_warmup: bool = True):
    compute_cfg = {
        "solver": {
            "public": {"use_warmup": use_warmup},
            "tuning": {
                "stopping": {"gtol": -1, "ftol": -1, "xtol": -1},
                "warmup": {"steps": 2},
                "lbfgsb": {"maxiter": 5},
            },
        },
        "passes": {"capture": {"every": 1, "mode": mode, "final_always": True}},
    }
    pack = make_solvepack_from_data_defaults(compute_cfg=compute_cfg)
    pack.actions = pack.actions[:1]
    pack.action_num = 1
    return pack


def test_mode_both_records_warmup_and_main():
    pack = _make_pack("both", use_warmup=True)
    vp = solve(pack)
    by_pass: dict[int, list] = {}
    for fr in vp.frames:
        by_pass.setdefault(fr.meta["pass_id"], []).append(fr)
    assert all(len(frames) >= 2 for frames in by_pass.values())


def test_mode_main_only_increases_frames_vs_none():
    pack_none = _make_pack("none", use_warmup=True)
    vp0 = solve(pack_none)
    pack_main = _make_pack("main", use_warmup=True)
    vp1 = solve(pack_main)
    assert len(vp1.frames) > len(vp0.frames)


def test_mode_none_only_final_frames():
    pack = _make_pack("none", use_warmup=True)
    vp = solve(pack)
    assert len(vp.frames) == pack.action_num
    assert all(fr.meta["frame_in_pass"] == "final" for fr in vp.frames)
