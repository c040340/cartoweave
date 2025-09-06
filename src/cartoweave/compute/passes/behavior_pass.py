from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import copy
import numpy as np

from cartoweave.contracts.solvepack import SolvePack
try:  # backward compatible type hints
    from cartoweave.contracts.solvepack import Behavior, LabelState  # type: ignore
except Exception:  # pragma: no cover - fallback for older contracts
    Behavior = Dict[str, Any]  # type: ignore
    LabelState = Dict[str, Any]  # type: ignore

from cartoweave.compute.geom.anchor import anchor_xy


@dataclass
class _State:
    labels: List[LabelState]
    active: np.ndarray  # (N,) bool


class BehaviorPass:
    def __init__(self, pack: SolvePack):
        labels_raw = pack.scene0.get("labels", [])
        WH = np.asarray(pack.scene0.get("WH", np.zeros((len(labels_raw), 2))), float)
        labels: List[LabelState] = []
        for i, lm in enumerate(labels_raw):
            anchor = lm.get("anchor")
            if anchor is None:
                anchor = {"kind": lm.get("anchor_kind"), "index": lm.get("anchor_index")}
            labels.append(
                {
                    "kind": lm.get("kind"),
                    "WH": WH[i].tolist() if i < len(WH) else None,
                    "anchor": anchor,
                    "meta": lm.get("meta", {}),
                }
            )
        self._state = _State(labels=copy.deepcopy(labels), active=pack.active_mask0.copy())
        self.scene0 = pack.scene0

    def begin_behavior(self, k: int, beh: Behavior, P_prev_star: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[LabelState], np.ndarray, Any]:
        P_k0 = np.asarray(P_prev_star, float).copy()
        labels = self._state.labels
        active = self._state.active
        ops = beh.get("ops", {}) or {}
        activate = ops.get("activate", [])
        deactivate = ops.get("deactivate", [])
        mutate = ops.get("mutate", [])

        beh_cfg = cfg.get("behavior", {}) if isinstance(cfg, dict) else {}
        place_first = bool(beh_cfg.get("place_on_first_activation"))
        snap_kind_change = bool(beh_cfg.get("snap_on_kind_change"))

        for idx in activate:
            if 0 <= idx < active.size:
                first = not bool(active[idx])
                active[idx] = True
                if first and place_first:
                    anc = labels[idx].get("anchor")
                    if anc:
                        xy = anchor_xy(anc, self.scene0)
                        if xy is not None:
                            P_k0[idx] = xy

        for idx in deactivate:
            if 0 <= idx < active.size:
                active[idx] = False

        for m in mutate:
            idx = m.get("id")
            if idx is None or not (0 <= idx < len(labels)):
                continue
            changes = m.get("set", {}) or {}
            lbl = labels[idx]
            if "WH" in changes and changes["WH"] is not None:
                lbl["WH"] = changes["WH"]
            if "anchor" in changes and changes["anchor"] is not None:
                lbl["anchor"] = changes["anchor"]
            if "meta" in changes and changes["meta"] is not None:
                lbl["meta"] = changes["meta"]
            if "kind" in changes:
                old_kind = lbl.get("kind")
                lbl["kind"] = changes["kind"]
                if snap_kind_change and changes["kind"] != old_kind:
                    anc = lbl.get("anchor")
                    if anc:
                        xy = anchor_xy(anc, self.scene0)
                        if xy is not None:
                            P_k0[idx] = xy

        self._state.labels = labels
        self._state.active = active
        scene_k = self.scene0
        return P_k0, copy.deepcopy(labels), active.copy(), scene_k

    def end_behavior(self, k: int, P_star: np.ndarray):
        return None
