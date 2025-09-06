from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve


def _gen_cfg(passes):
    return {
        "data": {
            "source": "generate",
            "generate": {"num_points": 3, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {"passes": passes},
        "behaviors": [{"solver": "lbfgs", "iters": 2}],
    }


def test_pass_manager_equivalence_smoke(as_legacy_pack):
    base = _gen_cfg({"capture": {"every": 1}})
    with_passes = _gen_cfg({"capture": {"every": 1}, "nan_guard": {}, "grad_clip": {"max_norm": 1e9}})

    sp1 = as_legacy_pack(build_solvepack_from_config(base, seed=1))
    sp2 = as_legacy_pack(build_solvepack_from_config(with_passes, seed=1))

    vp1 = solve(sp1)
    vp2 = solve(sp2)

    assert abs(vp1.last.E - vp2.last.E) < 1e-9
