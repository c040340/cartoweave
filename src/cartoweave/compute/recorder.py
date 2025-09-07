"""Runtime recorder collecting frames and events."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .types import Array2, Event, Frame, ViewPack


class Recorder:
    """Capture frames and events according to capture policy."""

    def __init__(self, pm, capture_cfg: dict[str, Any]):
        self.pm = pm
        self.every = max(1, int(capture_cfg.get("every", 1)))
        self.final_always = bool(capture_cfg.get("final_always", True))
        self.frames: list[Frame] = []
        self.events: list[Event] = []
        self.term_keys: set[str] = set()
        self._t0 = time.time()

    # ------------------------------------------------------------------
    def capture_stage_end(
        self,
        stage_id: int,
        p: Array2,
        labels: list[Any],
        metrics: dict[str, Any],
        mask: np.ndarray,
        *,
        final: bool = False,
    ) -> None:
        """Capture frame and stage event after a stage finishes."""

        n = p.shape[0]
        comps_raw = metrics.get("comps", {}) or {}
        self.term_keys.update(k for k, v in comps_raw.items() if np.asarray(v).shape == (n, 2))
        g_vec = metrics.get("G")
        gnorm = float(np.linalg.norm(g_vec)) if g_vec is not None and g_vec.size else 0.0
        if self.pm and isinstance(self.pm.cfg, dict):
            eps = (
                self.pm.cfg.get("solver", {})
                .get("internals", {})
                .get("stability", {})
                .get("eps_norm")
            )
            if eps is not None and gnorm < float(eps):
                gnorm = float(eps)

        want = (self.pm.eval_index % self.every) == 0
        if final and self.final_always:
            want = True
        if want:
            comps = {
                k: np.asarray(v, float)
                for k, v in comps_raw.items()
                if np.asarray(v).shape == (n, 2)
            }
            frame = Frame(
                P=np.asarray(p, float),
                comps=comps,
                E=float(metrics["E"]),
                Gnorm=gnorm,
                stage_id=stage_id,
                iter_id=self.pm.eval_index,
            )
            self.frames.append(frame)

        self.events.append(
            {
                "pass": "stage",
                "stage_id": stage_id,
                "global_iter": self.pm.eval_index,
                "info": "stage_end",
                "metrics": {"E": float(metrics["E"]), "Gnorm": gnorm},
            }
        )

    def capture_step_end(self, k: int, p: Array2, labels: list[Any], metrics: dict[str, Any]):
        mask = np.ones(p.shape[0], dtype=bool)
        self.capture_stage_end(k, p, labels, metrics, mask)

    def record_events(self, events: list[Event]) -> None:
        """Append events produced by passes."""

        if events:
            self.events.extend(events)

    # ------------------------------------------------------------------
    def to_viewpack(self, p_final: Array2, labels: list[Any]) -> ViewPack:
        """Convert captured data to a :class:`ViewPack`."""

        if not self.frames:
            # ensure at least one frame exists for ViewPack.last
            self.frames.append(
                Frame(
                    P=np.asarray(p_final, float),
                    comps={},
                    E=0.0,
                    Gnorm=0.0,
                    stage_id=-1,
                    iter_id=self.pm.eval_index,
                )
            )
        last = self.frames[-1]
        summary = {
            "frames_captured": len(self.frames),
            "terms_used": sorted(self.term_keys),
            "time_ms": int((time.time() - self._t0) * 1000),
            "global_iters": self.pm.eval_index,
        }
        return ViewPack(frames=self.frames, events=self.events, last=last, summary=summary)


__all__ = ["Recorder"]

