"""Minimal example using the config-driven data API."""

from cartoweave.compute import solve
from cartoweave.data.api import build_solvepack_from_config


def main():
    cfg = {
        "data": {
            "source": "generate",
            "generate": {
                "frame_size": [100, 100],
                "num_labels": 1,
                "num_points": 0,
                "num_lines": 0,
                "num_areas": 0,
                "num_steps": 1,
                "seed": 0,
            },
        }
    }
    sp = build_solvepack_from_config(cfg, seed=0)
    sp.validate()
    view = solve(sp)
    print("[minimal_solve] final positions", view.last.P)


if __name__ == "__main__":
    main()
