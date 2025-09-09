"""Runtime recorder collecting frames and events."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from .types import (
    Array2,
    Event,
    Frame,
    RecorderViewPack,
    ViewPack,
    VPFrame,
    VPPass,
    VPSources,
)


class Recorder:
    """Capture frames and events according to capture policy."""

    def __init__(self, pm, capture_cfg: dict[str, Any]):
        self.pm = pm
        self.every = max(1, int(capture_cfg.get("every", 1)))
        self.final_always = bool(capture_cfg.get("final_always", True))
        self.frames: list[Frame] = []
        self.events: list[Event] = []
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
    def to_viewpack(self, p_final: Array2, labels: list[Any]) -> RecorderViewPack:
        """Convert captured data to a :class:`RecorderViewPack`."""

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
        from .forces import enabled_terms, term_params_map

        compute_cfg = self.pm.cfg if isinstance(self.pm.cfg, dict) else {}
        pmap = term_params_map(compute_cfg)
        terms = enabled_terms(compute_cfg, phase="pre_anchor") + enabled_terms(
            compute_cfg, phase="anchor"
        )
        summary = {
            "frames_captured": len(self.frames),
            "terms_used": [
                {
                    "name": name,
                    "k_params": {
                        k: float(v)
                        for k, v in pmap.get(name, {}).items()
                        if k.startswith("k_") and v is not None
                    },
                }
                for name in terms
            ],
            "time_ms": int((time.time() - self._t0) * 1000),
            "global_iters": self.pm.eval_index,
            "pass_stats": self.pm.collect_stats(),
        }
        return RecorderViewPack(
            frames=self.frames, events=self.events, last=last, summary=summary
        )


class ViewRecorder:
    """Build a :class:`ViewPack` by recording full frames during optimization."""

    def __init__(self) -> None:
        self.N: int = 0
        self.labels: List[dict] = []
        self.WH: Optional[np.ndarray] = None
        self.sources: Optional[VPSources] = None
        self.defaults: Dict[str, Any] = {}
        self.aux: Dict[str, Any] = {}
        self.frames: List[VPFrame] = []
        self.passes: List[VPPass] = []
        self._current_pass: Optional[tuple[int, str, int]] = None

    # ------------------------------------------------------------------
    def start_run(
        self,
        N: int,
        labels: List[dict],
        WH: Optional[np.ndarray],
        sources: VPSources,
        defaults: Dict[str, Any],
        aux: Dict[str, Any],
    ) -> None:
        """Initialize run metadata before recording frames."""

        self.N = int(N)
        self.labels = labels
        self.WH = None if WH is None else np.asarray(WH, float)
        self.sources = sources
        self.defaults = defaults
        self.aux = aux
        self.frames = []
        self.passes = []
        self._current_pass = None

    # ------------------------------------------------------------------
    def start_pass(self, pass_id: int, pass_name: str) -> None:
        if self._current_pass is not None:
            raise RuntimeError("previous pass not closed")
        self._current_pass = (pass_id, pass_name, len(self.frames))

    def end_pass(self) -> None:
        if self._current_pass is None:
            raise RuntimeError("end_pass called without active pass")
        pass_id, pass_name, t_start = self._current_pass
        t_end = len(self.frames)
        self.passes.append(VPPass(pass_id, pass_name, t_start, t_end))
        self._current_pass = None

    # ------------------------------------------------------------------
    def record_frame(
        self,
        *,
        t: int,
        P_full: np.ndarray,
        comps_full: Dict[str, np.ndarray],
        E: float,
        active_mask: np.ndarray,
        anchors: Optional[np.ndarray],
        meta_base: Dict[str, Any],
        metrics: Dict[str, float],
        field: Optional[np.ndarray],
        G_snapshot: Optional[np.ndarray],
    ) -> None:
        if t != len(self.frames):
            raise ValueError(
                f"record_frame: expected t={len(self.frames)}, got {t}"
            )
        P = np.asarray(P_full, float)
        if P.shape != (self.N, 2):
            raise ValueError(f"P_full shape {P.shape} != ({self.N},2)")

        mask = np.asarray(active_mask, bool)
        if mask.shape != (self.N,):
            raise ValueError(
                f"active_mask shape {mask.shape} != ({self.N},)"
            )

        comps: Dict[str, np.ndarray] = {}
        for term, arr in comps_full.items():
            arr2 = np.asarray(arr, float)
            if arr2.shape != (self.N, 2):
                raise ValueError(
                    f"comps_full term '{term}' shape {arr2.shape} != ({self.N},2)"
                )
            arr2 = arr2.copy()
            arr2[~mask] = 0.0
            comps[term] = arr2

        meta = dict(meta_base)
        events: List[dict] = list(meta.pop("events", []))
        step_info = meta.pop("optimizer_step", None)
        if step_info is not None:
            events.append(
                {
                    "kind": "optimizer_step",
                    "algo": step_info.get("algo"),
                    "iter_in_algo": step_info.get("iter_in_algo"),
                    "step_size": step_info.get("step_size"),
                    "ls_evals": step_info.get("ls_evals"),
                    "wolfe": step_info.get("wolfe"),
                    "delta_E": step_info.get("delta_E"),
                    "gnorm": step_info.get("gnorm"),
                }
            )
        if not events:
            events.append(
                {"kind": "state_snapshot", "note": "no optimizer step this frame"}
            )

        meta["events"] = events
        meta["global_iter"] = t
        active_ids = np.flatnonzero(mask).tolist()
        meta["active_ids"] = active_ids
        meta["active_count"] = len(active_ids)
        if G_snapshot is not None:
            meta["G_snapshot"] = np.asarray(G_snapshot, float)

        anchors_arr = None
        if anchors is not None:
            anc = np.asarray(anchors, float)
            if anc.shape != (self.N, 2):
                raise ValueError(
                    f"anchors shape {anc.shape} != ({self.N},2)"
                )
            anchors_arr = anc

        frame = VPFrame(
            t=t,
            P=P,
            comps=comps,
            E=float(E),
            active_mask=mask,
            meta=meta,
            metrics=metrics,
            field=None if field is None else np.asarray(field, float),
            anchors=anchors_arr,
        )
        frame.validate(self.N)
        self.frames.append(frame)

    # ------------------------------------------------------------------
    def finish(self) -> ViewPack:
        if self._current_pass is not None:
            self.end_pass()
        assert self.sources is not None, "start_run must be called before finish"
        return ViewPack(
            schema_version="viewpack-v1",
            N=self.N,
            labels=self.labels,
            WH=self.WH,
            frames=self.frames,
            passes=self.passes,
            sources=self.sources,
            defaults=self.defaults,
            aux=self.aux,
        )


__all__ = ["Recorder", "ViewRecorder"]

