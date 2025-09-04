"""Lightweight logging helpers for engine utilities."""
from __future__ import annotations
from typing import Any

from ..utils.logging import logger as _logger


def log(user_logger: Any, msg: str) -> None:
    """Emit ``msg`` via ``user_logger`` if given or the global logger."""
    (user_logger or _logger).info(msg)


__all__ = ["log"]
