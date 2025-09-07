"""Helpers related to layout rendering."""
from __future__ import annotations
from typing import Any, Dict

def is_circle_label(lab: Dict) -> bool:
    """Return ``True`` if *lab* should be rendered as a circle.

    The viewer treats labels with ``mode == 'circle'`` specially – they are
    drawn as simple discs without a surrounding rectangle or text box and are
    excluded from the solver's active subset.
    """
    # ``lab`` may be a dataclass with a ``meta`` attribute or a plain dict.
    meta: Dict[str, Any]
    if isinstance(lab, dict):
        meta = lab.get("meta", lab)
    else:  # fall back to attribute lookup
        meta = getattr(lab, "meta", {})  # type: ignore[assignment]
    mode = meta.get("mode") if isinstance(meta, dict) else None
    return mode == "circle"


__all__ = ["is_circle_label"]
