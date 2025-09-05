# -*- coding: utf-8 -*-
"""CartoWeave public API (compute-only).

Engine & orchestrators were removed. Use compute.* directly or the exports below.
"""
from __future__ import annotations

from .compute.pack import SolvePack  # noqa: F401
from .compute.run import solve       # noqa: F401
from .version import __version__     # noqa: F401

__all__ = ["SolvePack", "solve", "__version__"]
