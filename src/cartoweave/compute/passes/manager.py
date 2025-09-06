from __future__ import annotations

from typing import Any, Callable

import numpy as np

from . import build_passes
from .base import Stage


class PassManager:
    """
    Central coordinator for compute passes.
    Responsibilities:
      - instantiate passes from spec (via build_passes)
      - let passes plan/mutate stages
      - wrap energy/grad chain
      - wrap step function (optional)
      - coordinate capture decisions and collect stats
    """
    def __init__(self, cfg: dict, passes_spec: list[Any] | None = None):
        self.cfg = cfg
        self.passes = build_passes(cfg, passes_spec)
        # runtime bookkeeping
        self.eval_index = 0
        self.stage_index = 0
        self.events: list[dict[str, Any]] = []
        for p in self.passes:
            p.pm = self  # type: ignore[attr-defined]
            name = getattr(p, "name", "") or p.__class__.__name__
            self.emit_event({
                "pass": name,
                "stage_id": -1,
                "global_iter": 0,
                "info": "init",
                "metrics": {},
            })

    # ---------- stages ----------
    def plan_stages(self, ctx: Any, base_stages: list[Any]) -> list[Any]:
        """Return planned stages after giving passes a chance to modify them."""

        stages = list(base_stages or [])
        # let passes create/adjust stages first (plan)
        for p in self.passes:
            plan = getattr(p, "plan_stages", None)
            if callable(plan):
                stages = plan(ctx, stages)
        # then give each pass a chance to mutate each stage
        mutate = [(p, getattr(p, "mutate_stage", None)) for p in self.passes]
        out: list[Any] = []
        for st in stages:
            st2 = st
            for _p, fn in mutate:
                if callable(fn):
                    st2 = fn(ctx, st2)
            out.append(st2)
        return out

    # ---------- wrapping ----------
    def wrap_energy(
        self,
        energy_fn: Callable[[np.ndarray, Any, Any, np.ndarray, dict[str, Any]], tuple[float, np.ndarray, dict[str, np.ndarray]]],
    ) -> Callable[[np.ndarray, Any, Any, np.ndarray, dict[str, Any]], tuple[float, np.ndarray, dict[str, np.ndarray]]]:
        """Wrap the energy/gradient entrypoint with pass-provided guards."""

        wrapped = energy_fn
        for p in self.passes:
            fn = getattr(p, "wrap_energy", None)
            if callable(fn):
                wrapped = fn(wrapped)

        def _tracked(p, labels, scene, active_mask, cfg):
            self.eval_index += 1
            p = np.asarray(p, float)
            active_mask = np.asarray(active_mask, bool)
            n = p.shape[0]
            if p.shape != (n, 2):
                raise ValueError(f"P shape {p.shape} expected ({n},2)")
            if active_mask.shape != (n,):
                raise ValueError(
                    f"active_mask shape {active_mask.shape} expected ({n},)"
                )

            e, g, comps = wrapped(p, labels, scene, active_mask, cfg)

            metrics: dict[str, Any] = {}

            if not np.isfinite(e):
                metrics["nonfinite_E"] = 1
                e = float(np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0))
            else:
                e = float(e)

            if g is None:
                g = np.zeros((n, 2), dtype=float)
            else:
                g = np.asarray(g, float)
                if g.shape != (n, 2):
                    raise ValueError(f"G shape {g.shape} expected ({n},2)")
                bad = ~np.isfinite(g)
                if bad.any():
                    metrics["nonfinite_G"] = int(bad.sum())
                    g = g.copy()
                    g[bad] = 0.0

            comps2: dict[str, np.ndarray] = {}
            bad_comp_total = 0
            for k, v in (comps or {}).items():
                v = np.asarray(v, float)
                if v.shape != (n, 2):
                    raise ValueError(
                        f"component '{k}' shape {v.shape} expected ({n},2)"
                    )
                bad = ~np.isfinite(v)
                if bad.any():
                    bad_comp_total += int(bad.sum())
                    v = v.copy()
                    v[bad] = 0.0
                comps2[k] = v
            if bad_comp_total:
                metrics["nonfinite_comp"] = bad_comp_total

            clip_cfg: dict[str, Any] = {}
            if isinstance(cfg, dict):
                internals = cfg.get("solver", {}).get("internals", {}) or {}
                clip_cfg = internals.get("clip", {}) or {}
            fam = clip_cfg.get("force_abs_max")
            if fam is not None:
                fam = float(fam)
                over = np.abs(g) > fam
                if over.any():
                    metrics["force_clip"] = int(over.sum())
                np.clip(g, -fam, fam, out=g)
                for k, v in comps2.items():
                    over = np.abs(v) > fam
                    if over.any():
                        metrics["force_clip"] = metrics.get("force_clip", 0) + int(over.sum())
                    np.clip(v, -fam, fam, out=comps2[k])

            eam = clip_cfg.get("energy_abs_max")
            if eam is not None:
                eam = float(eam)
                if abs(e) > eam:
                    metrics["energy_clip"] = 1
                    e = max(min(e, eam), -eam)

            if metrics:
                info = "nonfinite" if any(k.startswith("nonfinite") for k in metrics) else "clip"
                self.emit_event({"pass": "wrap_energy", "info": info, "metrics": metrics})

            return e, g, comps2

        return _tracked

    def wrap_step(self, step_fn: Callable[[np.ndarray, np.ndarray, dict[str, Any]], np.ndarray]) -> Callable[[np.ndarray, np.ndarray, dict[str, Any]], np.ndarray]:
        """
        step_fn: (P_old, P_proposed, iter_metrics) -> P_applied
        Default pipeline is identity; passes can clamp/modify step (e.g., StepLimitPass).
        """
        wrapped = step_fn
        for p in self.passes:
            fn = getattr(p, "wrap_step", None)
            if callable(fn):
                wrapped = fn(wrapped)

        def _stage_wrap(p_old, p_new, metrics):
            res = wrapped(p_old, p_new, metrics)
            self.stage_index += 1
            return res

        return _stage_wrap

    # ---------- capture ----------
    def want_capture(self, ctx: Any, eval_i: int, frames_len: int) -> bool:
        """Return True if any pass wants to capture the current eval."""

        for p in self.passes:
            fn = getattr(p, "want_capture", None)
            if callable(fn) and fn(ctx, eval_i, frames_len):
                return True
        return False

    # ---------- helpers ----------
    def stage_mask(self, active: np.ndarray, stage: Stage) -> np.ndarray:
        """Return active mask for a stage, validating shape."""

        mask = getattr(stage, "mask", None)
        if mask is None:
            return active
        mask_arr = np.asarray(mask, bool)
        if mask_arr.shape != active.shape:
            raise ValueError(
                f"stage mask shape mismatch: expected {active.shape}, got {mask_arr.shape}"
            )
        return mask_arr

    def build_recorder(self):
        """Instantiate a recorder using capture configuration."""

        from cartoweave.compute.recorder import Recorder

        cap_cfg = (
            self.cfg.get("passes", {}).get("capture", {}) if isinstance(self.cfg, dict) else {}
        )
        return Recorder(self, cap_cfg)

    # ---------- stats ----------
    def collect_stats(self) -> dict[str, Any]:
        """Collect optional statistics from passes."""

        out: dict[str, Any] = {}
        for p in self.passes:
            name = getattr(p, "name", "") or p.__class__.__name__
            stats = getattr(p, "stats", None)
            if stats:
                out[name] = stats
        return out

    # ---------- events ----------
    def emit_event(self, event: dict[str, Any]) -> None:
        """Queue an event emitted by a pass."""

        event.setdefault("stage_id", self.stage_index)
        event.setdefault("global_iter", self.eval_index)
        self.events.append(event)

    def pop_events(self) -> list[dict[str, Any]]:
        """Return and clear collected events."""

        evs = list(self.events)
        self.events.clear()
        return evs
