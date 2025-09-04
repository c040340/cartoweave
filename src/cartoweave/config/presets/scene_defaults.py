from __future__ import annotations
from typing import Dict, Any

def scene_defaults() -> Dict[str, Any]:
    """
    Generic scene/canvas defaults that many call-sites expect.
    These are intentionally conservative and engine-neutral.
    """
    return {
        "scene.width_px": 1080,
        "scene.height_px": 1920,
        "scene.dpi": 96,
        "scene.margin_px": 8,
    }
