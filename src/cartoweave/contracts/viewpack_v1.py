from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NotRequired, Optional, Tuple, TypedDict

import numpy as np


class FrameMeta(TypedDict):
    """Typed metadata required for each :class:`VPFrame`."""

    global_iter: int
    active_ids: List[int]
    active_count: int
    events: List[Dict[str, Any]]
    G_snapshot: NotRequired[np.ndarray]


@dataclass
class VPFrame:
    t: int
    P: np.ndarray
    comps: Dict[str, np.ndarray]
    E: float
    active_mask: np.ndarray
    meta: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    field: Optional[np.ndarray] = None
    anchors: Optional[np.ndarray] = None

    def validate(self, N: int) -> None:
        if self.P.shape != (N, 2):
            raise ValueError(f"frame t={self.t}: P shape {self.P.shape} != ({N},2)")

        # ADD: 先校验 active_mask，再基于 active 索引做有限性检查
        if self.active_mask.shape != (N,):
            raise ValueError(
                f"frame t={self.t}: active_mask shape {self.active_mask.shape} != ({N},)"
            )
        if self.active_mask.dtype != np.bool_:
            raise ValueError(f"frame t={self.t}: active_mask must be boolean")
        am = self.active_mask  # 便于后续使用

        # DELETE: 全量有限性检查（会拒绝未激活=NaN）
        # if not np.isfinite(self.P).all():
        #     raise ValueError(f"frame t={self.t}: P contains non-finite values")

        # ADD: 只检查激活元素为有限数；未激活允许为 NaN
        if not np.isfinite(self.P[am]).all():
            bad = np.where(~np.isfinite(self.P[am]))[0].tolist()
            raise ValueError(f"frame t={self.t}: P has non-finite values at active indices {bad}")

        meta = self.meta
        required_keys = ["global_iter", "active_ids", "active_count", "events"]
        for k in required_keys:
            if k not in meta:
                raise ValueError(f"frame t={self.t}: meta missing key '{k}'")

        if meta["global_iter"] != self.t:
            raise ValueError(
                f"frame t={self.t}: meta.global_iter {meta['global_iter']} != t"
            )

        active_ids = np.array(meta["active_ids"], dtype=int)
        if not np.array_equal(np.flatnonzero(self.active_mask), active_ids):
            raise ValueError(f"frame t={self.t}: active_ids mismatch")

        if int(meta["active_count"]) != int(self.active_mask.sum()):
            raise ValueError(f"frame t={self.t}: active_count mismatch")

        if not meta["events"]:
            raise ValueError(f"frame t={self.t}: events must not be empty")

        for term, arr in self.comps.items():
            if arr.shape != (N, 2):
                raise ValueError(
                    f"frame t={self.t}: comp '{term}' shape {arr.shape} != ({N},2)"
                )
            # DELETE: 全量有限性检查
            # if not np.isfinite(arr).all():
            #     raise ValueError(f"frame t={self.t}: comp '{term}' has non-finite values")

            # ADD: 仅对激活元素要求有限
            if not np.isfinite(arr[am]).all():
                bad = np.where(~np.isfinite(arr[am]))[0].tolist()
                raise ValueError(
                    f"frame t={self.t}: comp '{term}' has non-finite values at active indices {bad}"
                )

            # 保持：未激活处必须为零（既保证物理含义，也避免噪声）
            if np.any(arr[~am]):
                raise ValueError(
                    f"frame t={self.t}: comp '{term}' has nonzero forces where inactive"
                )

        if not np.isfinite(self.E):
            raise ValueError(f"frame t={self.t}: E must be finite")

        if self.anchors is not None:
            anc = np.asarray(self.anchors, dtype=float)
            if anc.shape != (N, 2):
                raise ValueError(
                    f"frame t={self.t}: anchors shape {anc.shape} != ({N},2)"
                )
            # Only enforce finiteness on active labels; inactive ones may hold NaNs
            if not np.isfinite(anc[am]).all():
                bad = np.where(~np.isfinite(anc[am]))[0].tolist()
                raise ValueError(
                    f"frame t={self.t}: anchors contain non-finite values at active indices {bad}"
                )

        if "G_snapshot" in meta:
            G_snap = np.asarray(meta["G_snapshot"], dtype=float)
            if G_snap.shape != (N, 2):
                raise ValueError(
                    f"frame t={self.t}: meta.G_snapshot invalid shape"
                )

            # DELETE: 全量有限性检查
            # if not np.isfinite(G_snap).all():
            #     raise ValueError(
            #         f"frame t={self.t}: meta.G_snapshot invalid shape or non-finite"
            #     )

            # ADD: 仅对激活元素要求有限
            if not np.isfinite(G_snap[am]).all():
                bad = np.where(~np.isfinite(G_snap[am]))[0].tolist()
                raise ValueError(
                    f"frame t={self.t}: meta.G_snapshot has non-finite values at active indices {bad}"
                )

            comps_sum = np.zeros((N, 2), dtype=float)
            for arr in self.comps.values():
                comps_sum += arr

            # DELETE: 全量守恒检查
            # if not np.allclose(G_snap, -comps_sum, atol=1e-6):
            #     raise ValueError(
            #         f"frame t={self.t}: G_snapshot not conserved with components"
            #     )

            # ADD: 只在激活元素上检查守恒
            if not np.allclose(G_snap[am], -comps_sum[am], atol=1e-6):
                raise ValueError(
                    f"frame t={self.t}: G_snapshot not conserved with components (active only)"
                )

        # DELETE: 全量 field 有限性检查
        # if self.field is not None and not np.isfinite(self.field).all():
        #     raise ValueError(f"frame t={self.t}: field contains non-finite values")

        # ADD: 如存在 field，仅对激活元素检查为有限
        if self.field is not None:
            # 允许未激活为 NaN；若 field 不是 (N,…) 结构，你可在此加形状断言
            if not np.isfinite(self.field[am]).all():
                raise ValueError(f"frame t={self.t}: field has non-finite values at active indices")


@dataclass
class VPPass:
    pass_id: int
    pass_name: str
    t_start: int
    t_end: int

    def validate(self) -> None:
        if self.t_start < 0 or self.t_end <= self.t_start:
            raise ValueError(
                f"pass {self.pass_id}: invalid time range ({self.t_start}, {self.t_end})"
            )


@dataclass
class VPSources:
    points: np.ndarray
    lines: List[np.ndarray]
    areas: List[dict]
    frame_size: Tuple[int, int]

    def validate(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"sources.points shape {self.points.shape} != (M,2)")
        if not np.isfinite(self.points).all():
            raise ValueError("sources.points contains non-finite values")

        for i, line in enumerate(self.lines):
            if line.ndim != 2 or line.shape[1] != 2:
                raise ValueError(f"sources.lines[{i}] invalid shape {line.shape}")
            if not np.isfinite(line).all():
                raise ValueError(f"sources.lines[{i}] contains non-finite values")

        for i, area in enumerate(self.areas):
            kind = area.get("kind")
            if kind == "poly":
                xy = np.asarray(area.get("xy"), dtype=float)
                if xy.ndim != 2 or xy.shape[1] != 2:
                    raise ValueError(
                        f"sources.areas[{i}]: polygon xy invalid shape {xy.shape}"
                    )
                if not np.isfinite(xy).all():
                    raise ValueError(
                        f"sources.areas[{i}]: polygon xy contains non-finite values"
                    )
            elif kind == "circle":
                xy = np.asarray(area.get("xy"), dtype=float)
                r = float(area.get("r", np.nan))
                if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(r):
                    raise ValueError(
                        f"sources.areas[{i}]: circle parameters invalid or non-finite"
                    )
            else:
                raise ValueError(f"sources.areas[{i}]: unknown kind '{kind}'")

        if len(self.frame_size) != 2:
            raise ValueError("sources.frame_size must be length-2")
        if not all(isinstance(v, int) and v > 0 for v in self.frame_size):
            raise ValueError("sources.frame_size must contain positive ints")


@dataclass
class ViewPack:
    schema_version: str
    N: int
    labels: List[dict]
    WH: Optional[np.ndarray]
    frames: List[VPFrame]
    passes: List[VPPass]
    sources: VPSources
    defaults: Dict[str, Any] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)

    _pass_lookup: Dict[int, VPPass] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.schema_version != "viewpack-v1":
            raise ValueError("schema_version must be 'viewpack-v1'")

        if len(self.labels) != self.N:
            raise ValueError("labels length does not match N")

        if self.WH is not None:
            if self.WH.shape != (self.N, 2):
                raise ValueError(f"WH shape {self.WH.shape} != ({self.N},2)")
            if not np.isfinite(self.WH).all():
                raise ValueError("WH contains non-finite values")

        self.sources.validate()

        # Validate frames
        expected_t = 0
        for frame in self.frames:
            if frame.t != expected_t:
                raise ValueError(
                    f"frames out of order: expected t={expected_t}, got {frame.t}"
                )
            frame.validate(self.N)
            expected_t += 1
        # Validate passes
        if self.frames:
            if not self.passes:
                raise ValueError("passes must cover frames but none provided")
            last_end = 0
            last_id = -1
            for p in self.passes:
                p.validate()
                if p.pass_id <= last_id:
                    raise ValueError("passes not sorted by pass_id")
                if p.t_start != last_end:
                    raise ValueError(
                        f"pass {p.pass_id}: t_start {p.t_start} does not follow previous end {last_end}"
                    )
                if p.t_end <= p.t_start:
                    raise ValueError(
                        f"pass {p.pass_id}: t_end {p.t_end} <= t_start {p.t_start}"
                    )
                last_end = p.t_end
                last_id = p.pass_id
            if last_end != len(self.frames):
                raise ValueError("passes do not cover all frames")
        else:
            if self.passes:
                raise ValueError("no frames but passes provided")

        self._pass_lookup = {p.pass_id: p for p in self.passes}

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def num_frames(self) -> int:
        return len(self.frames)

    def get_frame(self, t: int) -> VPFrame:
        if t < 0 or t >= len(self.frames):
            raise IndexError(f"frame index {t} out of range")
        return self.frames[t]

    def list_force_terms(self) -> List[str]:
        terms: set[str] = set()
        for frame in self.frames:
            terms.update(frame.comps.keys())
        return sorted(terms)

    def get_pass(self, pass_id: int) -> VPPass:
        try:
            return self._pass_lookup[pass_id]
        except KeyError:
            raise KeyError(f"pass_id {pass_id} not found") from None

    def find_frame_for_pass(self, pass_id: int, bias: str = "start") -> VPFrame:
        p = self.get_pass(pass_id)
        if bias == "start":
            t = p.t_start
        elif bias == "end":
            t = p.t_end - 1
        else:
            raise ValueError("bias must be 'start' or 'end'")
        return self.get_frame(t)

    def active_at(self, t: int) -> np.ndarray:
        frame = self.get_frame(t)
        return np.flatnonzero(frame.active_mask)

    def force_at(self, t: int, term: str) -> np.ndarray:
        frame = self.get_frame(t)
        try:
            return frame.comps[term]
        except KeyError:
            raise KeyError(f"term '{term}' not found in frame {t}") from None
