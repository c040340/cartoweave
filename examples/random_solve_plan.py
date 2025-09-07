"""Minimal example that generates a scene using data defaults."""
from __future__ import annotations

from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.logging import init_logging_from_cfg


def main() -> None:
    sp = make_solvepack_from_data_defaults()
    sp.validate()
    init_logging_from_cfg(getattr(sp, "cfg", None))
    scene = sp.scene0
    print("frame", scene.frame_size)
    print("labels", sp.L)


if __name__ == "__main__":
    main()
