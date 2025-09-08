"""Dump merged compute defaults for inspection."""
from __future__ import annotations

import json

from cartoweave.config.loader import load_compute_config


def main() -> None:
    cfg = load_compute_config()["compute"]
    print(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
