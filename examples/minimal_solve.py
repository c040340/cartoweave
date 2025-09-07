"""Minimal example using the default data configuration."""

from cartoweave.compute import solve
from cartoweave.data.api import make_solvepack_from_data_defaults


def main() -> None:
    sp = make_solvepack_from_data_defaults(
        compute_cfg={"public": {"forces": {"anchor.spring": {"enable": True, "k": 1.0}}}}
    )
    sp.validate()
    view = solve(sp)
    print("[minimal_solve] final positions", view.last.P)


if __name__ == "__main__":
    main()
