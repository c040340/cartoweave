from __future__ import annotations

import warnings
from typing import Any, Dict, List

from .scene_script_runner import run_scene_script


def run_timeline(
    scene: Dict[str, Any],
    timeline: Dict[str, Any] | List[Dict[str, Any]],
    solve_plan: Dict[str, Any] | List[Dict[str, Any]],
    cfg: Dict[str, Any] | None = None,
):
    """Deprecated wrapper forwarding to :func:`run_scene_script`.

    Parameters mirror the old API but a ``DeprecationWarning`` is emitted and the
    implementation simply delegates to :func:`run_scene_script`.
    """

    warnings.warn(
        "timeline_runner is deprecated, use scene_script_runner",
        DeprecationWarning,
        stacklevel=2,
    )
    cfg = cfg or {}
    steps = timeline.get("steps") if isinstance(timeline, dict) else timeline
    return run_scene_script(scene, steps, solve_plan, cfg)
