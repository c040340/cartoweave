"""Minimal example that generates a scene using the new API."""
from __future__ import annotations

import numpy as np
from cartoweave.data.scene.make import get_scene


def main() -> None:
    scene = get_scene((200, 100), seed=0)
    print("points", scene["points"].shape)
    print("lines", len(scene["lines"]))
    print("areas", len(scene["areas"]))
    print("labels", scene["labels_init"].shape[0])


if __name__ == "__main__":
    main()
