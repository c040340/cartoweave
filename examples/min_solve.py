"""Run a minimal solve using the default configuration."""

from cartoweave.compute import solve
from cartoweave.data.api import make_solvepack_from_data_defaults

sp = make_solvepack_from_data_defaults(
    compute_cfg={
        "weights": {"anchor.spring": 1.0},
        "passes": {"capture": {"every": 1, "final_always": True}},
    }
)
sp.validate()
view = solve(sp)
print(
    "frames:", view.summary.get("frames_captured"),
    "E_last:", view.summary.get("E_last"),
)
