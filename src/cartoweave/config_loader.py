"""Loading configuration files with a strict schema.
"""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------

def _parse_dataclass(cls, data: Any, path: Tuple[str, ...] = ()):
    """Recursively parse ``data`` into dataclass ``cls``.

    Raises ``KeyError`` if unknown keys are present or required keys missing.
    """
    if not is_dataclass(cls):
        # Basic types just return as is (lists handled separately)
        if getattr(cls, "__origin__", None) is list and isinstance(data, list):
            item_type = cls.__args__[0]
            return [
                _parse_dataclass(item_type, v, path + (str(i),)) if is_dataclass(item_type) else v
                for i, v in enumerate(data)
            ]
        return data

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {cls.__name__} at {'.'.join(path)}")

    hints = get_type_hints(cls)
    allowed = set(hints.keys())
    extra = set(data) - allowed
    if extra:
        raise KeyError(f"Unknown keys {extra} at {'.'.join(path) or '<root>'}")

    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            raise KeyError(f"Missing key '{f.name}' at {'.'.join(path) or '<root>'}")
        ft = hints[f.name]
        kwargs[f.name] = _parse_dataclass(ft, data[f.name], path + (f.name,))
    return cls(**kwargs)  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Core config models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Experiment:
    regen_data: bool
    seed: int
    num_points: int
    num_lines: int
    num_areas: int
    num_actions: int

@dataclass(frozen=True)
class SolverQuickInit:
    strategy: str
    perturb_px: float
    clamp_to_frame: bool

@dataclass(frozen=True)
class SolverQuick:
    mode: str
    max_iters: int
    init: SolverQuickInit

@dataclass(frozen=True)
class Stopping:
    gtol_ref_kind: str
    gtol_abs: float
    gtol_rel: float
    gtol_cap: Optional[float]

@dataclass(frozen=True)
class Lbfgsb:
    restarts: int
    m: int
    factr: float
    maxiter: int
    maxfun: int
    maxls: int
    disp: bool

@dataclass(frozen=True)
class Warmup:
    steps: int
    pixel_equiv: float
    step_cap: float

@dataclass(frozen=True)
class Retry:
    enable: bool
    keep_best: bool
    noise_px: float
    noise_ratio: float
    noise_seed: int

@dataclass(frozen=True)
class StepControl:
    max_step_px: float
    frac_of_rect: float
    backtrack_trials: int

@dataclass(frozen=True)
class Prox:
    enable: bool
    mu: float
    decay: float
    on_events_only: bool

@dataclass(frozen=True)
class Acceptance:
    monotone: bool
    try_pg_cooldown: int

@dataclass(frozen=True)
class RobustTerm:
    name: str
    cap_p95_mult: float

@dataclass(frozen=True)
class RobustForces:
    enable: bool
    terms: List[RobustTerm]

@dataclass(frozen=True)
class Watchdog:
    enable: bool
    window: int
    rel_F_drop: float
    min_switch_iters: int
    perturb_px: float

@dataclass(frozen=True)
class SolverTuning:
    stopping: Stopping
    lbfgsb: Lbfgsb
    warmup: Warmup
    retry: Retry
    step_control: StepControl
    prox: Prox
    acceptance: Acceptance
    cooldown_pg_steps: int
    robust_forces: RobustForces
    watchdog: Watchdog

@dataclass(frozen=True)
class Solver:
    quick: SolverQuick
    tuning: SolverTuning

# Physics -----------------------------------------------------------------

@dataclass(frozen=True)
class QuickPointSpring:
    rest_mode: str
    rest_offset_px: float

@dataclass(frozen=True)
class QuickLineSpring:
    rest_mode: str
    edge_follow_span_pct: float

@dataclass(frozen=True)
class PhysicsQuick:
    point_spring: QuickPointSpring
    line_spring: QuickLineSpring

@dataclass(frozen=True)
class Enabled:
    label_label_repulsion: bool
    boundary: bool
    point_spring: bool
    line_spring: bool
    area_softout: bool
    area_cross: bool
    anchor: bool
    focus: bool

@dataclass(frozen=True)
class FocusFalloff:
    beta: float
    decay: float

@dataclass(frozen=True)
class Focus:
    shape: str
    center_norm: List[float]
    axes_px: List[float]
    angle_deg: float
    falloff: FocusFalloff

@dataclass(frozen=True)
class PointSpring:
    k: float
    rest_length_px: float
    tangent_damping: float

@dataclass(frozen=True)
class LineSpring:
    k: float
    normal_spring_k: float
    tangent_damping: float

@dataclass(frozen=True)
class LabelLabelRepulsion:
    k_rr: float
    softmin_beta: float
    distance_model: str

@dataclass(frozen=True)
class Boundary:
    k_boundary: float
    inverse_square_eps: float
    outside_penalty: float

@dataclass(frozen=True)
class AreaSoftout:
    k: float
    width_px: float
    decay: float

@dataclass(frozen=True)
class AreaCross:
    k: float
    width_px: float
    decay: float

@dataclass(frozen=True)
class Anchor:
    k_anchor_spring: float
    falloff: float
    clip: float

@dataclass(frozen=True)
class PhysicsTuning:
    enabled: Enabled
    focus: Focus
    point_spring: PointSpring
    line_spring: LineSpring
    label_label_repulsion: LabelLabelRepulsion
    boundary: Boundary
    area_softout: AreaSoftout
    area_cross: AreaCross
    anchor: Anchor

@dataclass(frozen=True)
class KMultipliers:
    boundary: float
    label_label_repulsion: float
    anchor: float

@dataclass(frozen=True)
class Scaling:
    enable_autocal: bool
    base_term: str
    length_ref_px: float
    energy_ref: float
    sigma_scale: float
    k_multipliers: KMultipliers

@dataclass(frozen=True)
class Physics:
    quick: PhysicsQuick
    tuning: PhysicsTuning
    scaling: Scaling

# Numerics / Logging -------------------------------------------------------

@dataclass(frozen=True)
class Numerics:
    eps_div: float
    eps_sqrt: float
    clamp_limits: List[float]
    finite_check: str

@dataclass(frozen=True)
class LoggingDebug:
    level: str
    debug_solver: bool
    debug_check: bool
    emit_grad_check: bool
    emit_topk_forces: bool
    dump_active_ids: bool
    profile_timers: bool

@dataclass(frozen=True)
class CoreConfig:
    experiment: Experiment
    solver: Solver
    physics: Physics
    numerics: Numerics
    logging_debug: LoggingDebug

# Viz config ---------------------------------------------------------------

@dataclass(frozen=True)
class VizLayout:
    grid: List[int]
    aspect_mode: str

@dataclass(frozen=True)
class VizExportVideo:
    enable: bool
    fps: int
    codec: str

@dataclass(frozen=True)
class VizExport:
    out_dir: str
    video: VizExportVideo

@dataclass(frozen=True)
class VizConfig:
    enable: bool
    export_stride: int
    figure_size: List[int]
    dpi: int
    panels: Dict[str, bool]
    layers: Dict[str, bool]
    layout: VizLayout
    axes: Dict[str, bool]
    theme: str
    colors: Dict[str, str]
    alpha: Dict[str, float]
    zorder: Dict[str, int]
    force_vectors: Dict[str, Any]
    force_breakdown: Dict[str, Any]
    surface3d: Dict[str, Any]
    timeline: Dict[str, Any]
    picker: Dict[str, Any]
    export: VizExport
    numerics: Dict[str, float]

# Bundle ------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigBundle:
    core: CoreConfig
    viz: VizConfig

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _merge(base: Dict[str, Any], override: Dict[str, Any], path: Tuple[str, ...], diffs: List[str]):
    for k, v in override.items():
        if k not in base:
            raise KeyError(f"Unknown override key {'.'.join(path + (k,))}")
        if isinstance(v, dict) and isinstance(base[k], dict):
            _merge(base[k], v, path + (k,), diffs)
        else:
            old = base[k]
            base[k] = v
            diffs.append(f"{'.'.join(path + (k,))}: {old} -> {v}")


def load_configs(config_path: str = "config.yaml", viz_path: str = "viz.yaml", run_path: str = "run.yaml") -> ConfigBundle:
    """Load config.yaml and viz.yaml, then shallow-merge optional run.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        core_dict = yaml.safe_load(f) or {}
    with open(viz_path, "r", encoding="utf-8") as f:
        viz_dict = yaml.safe_load(f) or {}

    diffs: List[str] = []
    if os.path.exists(run_path):
        with open(run_path, "r", encoding="utf-8") as f:
            run_dict = yaml.safe_load(f) or {}
        for k, v in run_dict.items():
            if k == "viz":
                if not isinstance(v, dict):
                    raise TypeError("run.yaml viz override must be mapping")
                _merge(viz_dict, v, ("viz",), diffs)
            else:
                if k not in core_dict:
                    raise KeyError(f"Unknown override key {k}")
                if isinstance(v, dict) and isinstance(core_dict[k], dict):
                    _merge(core_dict[k], v, (k,), diffs)
                else:
                    old = core_dict[k]
                    core_dict[k] = v
                    diffs.append(f"{k}: {old} -> {v}")

    core_cfg = _parse_dataclass(CoreConfig, core_dict)
    viz_cfg = _parse_dataclass(VizConfig, viz_dict)

    if diffs:
        print("Override diff:")
        for d in diffs:
            print("  ", d)

    return ConfigBundle(core_cfg, viz_cfg)

__all__ = ["load_configs", "ConfigBundle"]

