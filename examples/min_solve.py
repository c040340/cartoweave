"""Run a minimal solve using the default configuration."""

from cartoweave.compute import solve
from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.logging import init_logging_from_cfg

sp = make_solvepack_from_data_defaults(
    compute_cfg={
        "public": {"forces": {"anchor.spring": {"enable": True, "k_local": 1.0}}},
        "passes": {"capture": {"every": 1, "final_always": True}},
    }
)
sp.validate()
init_logging_from_cfg(getattr(sp, "cfg", None))
view = solve(sp)
final_E = view.frames[-1].E if view.frames else None
print("frames:", view.num_frames(), "E_last:", final_E)
