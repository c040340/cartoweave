# -*- coding: utf-8 -*-
"""Minimal example using the new data API."""

from cartoweave.data.api import build_solvepack_direct
from cartoweave.compute.run import solve


def main():
    sp = build_solvepack_direct(
        frame_size=(100, 100),
        n_labels=1,
        steps={"kind": "none"},
        seed=0,
        solver_cfg={"compute": {"weights": {"anchor.spring": 1.0}, "eps": {"numeric": 1e-12}}},
    )
    view = solve(sp)
    print("[minimal_solve] final positions", view.last.P)


if __name__ == "__main__":
    main()
