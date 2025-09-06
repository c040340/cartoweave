from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from cartoweave.contracts.solvepack import (
    SolvePack,
    LabelState,
    AnchorSpec,
    Behavior,
)
from cartoweave.compute.geom.anchor import anchor_xy


@dataclass
class RuntimeState:
    P: np.ndarray
    active: np.ndarray
    labels: List[LabelState]


def copy_label(lbl: LabelState) -> LabelState:
    return LabelState(
        kind=str(lbl.kind),
        WH=np.asarray(lbl.WH, float).copy(),
        anchor=AnchorSpec(lbl.anchor.kind, int(lbl.anchor.index)) if getattr(lbl, "anchor", None) else None,
        meta=dict(getattr(lbl, "meta", {}) or {}),
    )


def apply_behavior_step(pack: SolvePack, state: RuntimeState, behavior: Behavior) -> RuntimeState:
    cfg = dict(pack.cfg.get("behavior", {}))
    place_on_first = bool(cfg.get("place_on_first_activation", True))
    snap_on_kind_change = bool(cfg.get("snap_on_kind_change", False))
    line_mode = cfg.get("line_anchor_mode", "midpoint")
    area_mode = cfg.get("area_anchor_mode", "centroid")

    ops: Dict[str, Any] = behavior.ops if isinstance(behavior.ops, dict) else {}
    acts = list(ops.get("activate", []) or [])
    deact = list(ops.get("deactivate", []) or [])
    muts = list(ops.get("mutate", []) or [])

    for i in acts:
        if not state.active[i]:
            state.active[i] = True
            if place_on_first:
                anc = state.labels[i].anchor
                xy = anchor_xy(pack.scene0, anc, line_mode, area_mode) if anc else None
                if xy is not None:
                    state.P[i] = np.asarray(xy, float)

    for m in muts:
        i = int(m.get("id"))
        setv = dict(m.get("set", {}) or {})
        lbl = state.labels[i]
        prev_kind = lbl.kind

        if "kind" in setv:
            lbl.kind = str(setv["kind"])
        if "WH" in setv:
            wh = np.asarray(setv["WH"], float)
            if wh.shape == (2,):
                lbl.WH = wh
        if "anchor" in setv:
            a = setv["anchor"]
            if isinstance(a, dict) and "kind" in a and "index" in a:
                lbl.anchor = AnchorSpec(str(a["kind"]), int(a["index"]))
        if "meta" in setv and isinstance(setv["meta"], dict):
            lbl.meta.update(setv["meta"])

        if snap_on_kind_change and lbl.kind != prev_kind and lbl.anchor is not None:
            xy = anchor_xy(pack.scene0, lbl.anchor, line_mode, area_mode)
            if xy is not None:
                state.P[i] = np.asarray(xy, float)

    for i in deact:
        state.active[i] = False

    return state


class BehaviorPass:
    """Compatibility wrapper for legacy tests.

    This mimics the old ``begin_behavior``/``end_behavior`` API using the new
    :class:`RuntimeState` utilities.
    """

    def __init__(self, pack: SolvePack):
        self.pack = pack
        P = np.asarray(pack.P0, float).copy()
        active = np.asarray(pack.active0, bool).copy()
        labels = [copy_label(l) for l in pack.labels0]
        self.state = RuntimeState(P=P, active=active, labels=labels)

    def begin_behavior(self, k: int, beh: Behavior, P_prev_star: np.ndarray, cfg: Dict[str, Any]):
        self.state.P = np.asarray(P_prev_star, float).copy()
        self.state = apply_behavior_step(self.pack, self.state, beh)
        return (
            self.state.P.copy(),
            [copy_label(l) for l in self.state.labels],
            self.state.active.copy(),
            self.pack.scene0,
        )

    def end_behavior(self, k: int, P_star: np.ndarray):
        self.state.P = np.asarray(P_star, float).copy()
        return None


__all__ = [
    "RuntimeState",
    "copy_label",
    "apply_behavior_step",
    "BehaviorPass",
]

