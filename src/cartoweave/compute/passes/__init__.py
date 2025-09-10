# -*- coding: utf-8 -*-
"""Pass registry, pass builder and step-wise ``PassManager``."""

from __future__ import annotations
from typing import Any, Dict, List, Union
from .base import Context, ComputePass
from cartoweave.utils.logging import logger


def get_pass_cfg(cfg: dict, name: str, defaults: dict | None = None) -> dict:
    d = (cfg.get("passes", {}).get(name) or {})
    if defaults:
        out = dict(defaults)
        out.update(d)
        return out
    return dict(d)

from .capture import CapturePass
from .nan_guard import NaNGuardPass
from .grad_clip import GradClipPass
from .step_limit import StepLimitPass
from .geom_preproc import GeomPreprocPass

REGISTRY = {
    "nan_guard":  (NaNGuardPass, {"on_nan": "zero", "on_inf": "clip"}),
    "grad_clip":  (GradClipPass, {"max_norm": None, "max_abs": None}),
    "step_limit": (StepLimitPass, {"max_step_norm": 1.5}),
    "capture":    (CapturePass, {"every": 1, "final_always": True}),
    "geom_preproc": (GeomPreprocPass, {"tiny_eps": 1e-9}),
}


def build_passes(cfg: Dict, cfg_list: List[Union[str, Dict]] | None) -> List[ComputePass]:
    """Instantiate passes from specification.

    ``cfg_list`` may contain strings or ``{"name": ..., "args": ...}``
    dictionaries. ``CapturePass`` is added with defaults from ``cfg`` if not
    specified. ``geom_preproc`` and ``step_limit`` are inserted with defaults
    if missing.
    """
    passes: List[ComputePass] = []
    names = set()

    def _mk(name: str, args: Dict | None):
        entry = REGISTRY.get(name)
        if not entry:
            raise ValueError(f"Unknown pass: {name}")
        cls, defaults = entry
        kwargs = dict(defaults or {})
        if args:
            kwargs.update(args)
        inst = cls(**kwargs)
        passes.append(inst)
        names.add(name)

    cfg_list = cfg_list or []
    for item in cfg_list:
        if isinstance(item, str):
            _mk(item, None)
        elif isinstance(item, dict):
            _mk(item.get("name"), item.get("args"))
        else:
            raise ValueError("pass config must be str or dict")

    if "geom_preproc" not in names:
        cls, defaults = REGISTRY["geom_preproc"]
        passes.insert(0, cls(**defaults))
        names.add("geom_preproc")
    if "nan_guard" not in names:
        cls, defaults = REGISTRY["nan_guard"]
        passes.append(cls(**defaults))
        names.add("nan_guard")
    if "grad_clip" not in names:
        cls, defaults = REGISTRY["grad_clip"]
        passes.append(cls(**defaults))
        names.add("grad_clip")
    if "step_limit" not in names:
        cls, defaults = REGISTRY["step_limit"]
        passes.append(cls(**defaults))
        names.add("step_limit")
    if "capture" not in names:
        cls, defaults = REGISTRY["capture"]
        passes.append(cls(**defaults))
    return passes


DEFAULT_PIPELINE = ["calibration", "action", "forces", "solver", "capture"]


class PassManager:
    def __init__(
        self,
        cfg: dict | None = None,
        passes_spec: list[Any] | None = None,
        pipeline: list[str] | None = None,
    ) -> None:
        self.cfg = cfg or {}
        self.passes = build_passes(self.cfg, passes_spec)
        self.pass_map: dict[str, ComputePass] = {}
        self.pipeline = list(pipeline or DEFAULT_PIPELINE)
        self.eval_index = 0
        self.events: list[dict[str, Any]] = []
        for p in self.passes:
            p.pm = self  # type: ignore[attr-defined]
            name = getattr(p, "name", "") or p.__class__.__name__
            self.pass_map[name.lower()] = p
            self.emit_event({
                "pass": name,
                "global_iter": 0,
                "info": "init",
                "metrics": {},
            })
        logger.info(
            "PassManager init passes=%s pipeline=%s",
            [getattr(p, "name", "") or p.__class__.__name__ for p in self.passes],
            self.pipeline,
        )

    # ----- pipeline helpers -------------------------------------------------
    def ensure_pass(self, name: str, position: int | None = None):
        if name in self.pipeline:
            return
        if position is None:
            self.pipeline.insert(0, name)
        else:
            self.pipeline.insert(int(position), name)

    def remove_pass(self, name: str):
        self.pipeline = [p for p in self.pipeline if p != name]

    def run_single_pass(self, name: str, ctx: dict):
        p = self.pass_map.get(name)
        if p is None:
            entry = REGISTRY.get(name)
            if entry:
                cls, defaults = entry
                p = cls(**defaults)
                p.pm = self  # type: ignore[attr-defined]
                self.passes.append(p)
                self.pass_map[name] = p
        if p is not None:
            fn = getattr(p, "run", None)
            if callable(fn):
                fn(ctx)

    def run_step(self, ctx: dict):
        for name in self.pipeline:
            self.run_single_pass(name, ctx)

    # ----- wrapping ---------------------------------------------------------
    def wrap_energy(
        self,
        energy_fn,
    ):
        import numpy as np

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

            if metrics:
                info = (
                    "nonfinite" if any(k.startswith("nonfinite") for k in metrics) else "clip"
                )
                self.emit_event({"pass": "wrap_energy", "info": info, "metrics": metrics})

            return e, g, comps2

        return _tracked

    def wrap_step(self, step_fn):
        wrapped = step_fn
        for p in self.passes:
            fn = getattr(p, "wrap_step", None)
            if callable(fn):
                wrapped = fn(wrapped)
        return wrapped

    # ----- capture ----------------------------------------------------------
    def want_capture(self, ctx: dict, eval_i: int, frames_len: int) -> bool:
        for p in self.passes:
            fn = getattr(p, "want_capture", None)
            if callable(fn) and fn(ctx, eval_i, frames_len):
                return True
        return False

    def build_recorder(self):
        from cartoweave.compute.recorder import Recorder

        cap_cfg = (
            self.cfg.get("passes", {}).get("capture", {}) if isinstance(self.cfg, dict) else {}
        )
        return Recorder(self, cap_cfg)

    # ----- stats & events ---------------------------------------------------
    def collect_stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in self.passes:
            name = getattr(p, "name", "") or p.__class__.__name__
            stats = getattr(p, "stats", None)
            if stats:
                out[name] = stats
        return out

    def emit_event(self, event: dict[str, Any]) -> None:
        event.setdefault("global_iter", self.eval_index)
        self.events.append(event)
        logger.debug("pass_event %s", event)

    def pop_events(self) -> list[dict[str, Any]]:
        evs = list(self.events)
        self.events.clear()
        return evs


# ensure action pass registers itself
from .action_pass import ActionPass as _  # noqa: F401
# ensure calibration pass registers itself
from .calibration import CalibrationPass as _  # noqa: F401
# ensure label relax pass registers itself
from .label_relax_pass import LabelRelaxPass as _  # noqa: F401


__all__ = [
    "Context",
    "ComputePass",
    "PassManager",
    "build_passes",
    "REGISTRY",
    "get_pass_cfg",
]
