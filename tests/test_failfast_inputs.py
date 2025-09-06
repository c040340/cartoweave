import numpy as np
import pytest
from cartoweave.data.api import build_solvepack_from_config


def _cfg(n):
    return {
        "data": {
            "source": "generate",
            "generate": {"num_points": n, "num_lines": 0, "num_areas": 0, "num_steps": 1},
        },
        "compute": {"passes": {"capture": {"every": 1}}},
        "behaviors": [{"solver": "lbfgs", "iters": 1}],
    }


def test_validate_shapes():
    sp = build_solvepack_from_config(_cfg(3), seed=0)
    sp.validate()

    sp_bad = build_solvepack_from_config(_cfg(3), seed=0)
    sp_bad.P0 = np.zeros((sp_bad.L, 3))
    with pytest.raises(ValueError):
        sp_bad.validate()

    sp_bad2 = build_solvepack_from_config(_cfg(3), seed=0)
    sp_bad2.active0 = np.zeros((sp_bad2.L, 2), dtype=bool)
    with pytest.raises(ValueError):
        sp_bad2.validate()
