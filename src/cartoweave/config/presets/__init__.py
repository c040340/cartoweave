"""Presets package exposing legacy helpers and new base/scene defaults."""
from __future__ import annotations
import importlib.util
import pathlib
from types import ModuleType
from typing import List

# Load legacy presets from sibling file `presets.py` if it exists.
_legacy_module: ModuleType | None = None
# `__file__` points to `config/presets/__init__.py`; legacy module lives one
# directory up as `config/presets.py`.
_legacy_path = pathlib.Path(__file__).resolve().parent.parent / "presets.py"
if _legacy_path.exists():
    spec = importlib.util.spec_from_file_location(
        "cartoweave.config._legacy_presets", _legacy_path
    )
    if spec and spec.loader:
        _legacy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_legacy_module)  # type: ignore[misc]

if _legacy_module is not None:
    # Re-export public symbols from legacy module for backward compatibility
    for _name in dir(_legacy_module):
        if _name.startswith("_"):
            continue
        globals()[_name] = getattr(_legacy_module, _name)

    __all__: List[str] = [
        name for name in dir(_legacy_module) if not name.startswith("_")
    ]
else:
    __all__ = []
