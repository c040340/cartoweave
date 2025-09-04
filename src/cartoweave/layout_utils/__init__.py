"""Helpers related to layout rendering."""
from __future__ import annotations
from typing import Dict

def is_circle_label(lab: Dict) -> bool:
    """Return ``True`` if *lab* should be rendered as a circle.

    The viewer treats labels with ``mode == 'circle'`` specially – they are
    drawn as simple discs without a surrounding rectangle or text box and are
    excluded from the solver's active subset.
    """
    return lab.get("mode") == "circle"


__all__ = ["is_circle_label"]
