from __future__ import annotations

import logging
from typing import Optional, Mapping, Any

_LEVEL_MAP = {
    "none": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def _normalize(level: Optional[str]) -> int:
    if not level:
        return logging.WARNING
    return _LEVEL_MAP.get(str(level).strip().lower(), logging.WARNING)


def level_from_cfg(cfg: Optional[Mapping[str, Any]]) -> int:
    """Return default logging level (legacy config key removed)."""
    return logging.WARNING


def init_logging(level: int | str | None = None) -> None:
    """
    初始化根 logger 与 cartoweave 命名空间 logger；避免重复 handler。
    多次调用允许覆盖等级，但不重复添加 handler。
    """
    lvl = _normalize(level) if isinstance(level, str) or level is None else int(level)

    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(h)
    root.setLevel(lvl)

    logging.getLogger("cartoweave").setLevel(lvl)


def init_logging_from_cfg(cfg: Optional[Mapping[str, Any]]) -> None:
    init_logging(level_from_cfg(cfg))

