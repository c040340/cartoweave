"""Minimal example using the default data configuration."""

from cartoweave.compute import solve
from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.logging import init_logging_from_cfg


def main() -> None:
    sp = make_solvepack_from_data_defaults(
        compute_cfg={"public": {"forces": {"anchor.spring": {"enable": True, "k": 1.0}}}}
    )
    sp.validate()
    init_logging_from_cfg(getattr(sp, "cfg", None))
    view = solve(sp)
    if view.frames:
        print("[minimal_solve] final positions", view.frames[-1].P)
    else:
        print("[minimal_solve] no frames")


if __name__ == "__main__":
    main()
