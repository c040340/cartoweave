"""Public compute API."""
from . import init  # noqa: F401

from . import init as _init  # noqa: F401  # side effect: logging
from .solve import solve

__all__ = ["solve"]
