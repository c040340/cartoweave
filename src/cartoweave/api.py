from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import copy

import yaml
import numpy as np

# === 根据工程真实路径修正导入 ===
from cartoweave.compute.solve import solve as _run_solver
from cartoweave.labels import anchor_xy as _anchor_xy

Number = Union[int, float]


@dataclass
class SolveResult:
    coords: np.ndarray            # (T, N, 2)
    viewpack: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None


# internal helper classes -----------------------------------------------------

class _Label(dict):
    """Minimal label object with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:  # pragma: no cover - defensive
            raise AttributeError(name) from e

    def __setattr__(self, name, value) -> None:  # pragma: no cover - defensive
        self[name] = value


@dataclass
class _Action:
    """Internal action representation with explicit id fields."""

    label_id: int
    t: float
    type: str
    WH_to: Optional[Sequence[Number]] = None
    kind_to: Optional[str] = None
    action_id: Optional[int] = None

    # compat: some downstream passes still access ``id``
    @property
    def id(self) -> int:  # pragma: no cover - backward compat
        return self.label_id


@dataclass
class _Pack:
    L: int
    P0: List[Sequence[Number]]
    labels0: List[_Label]
    active0: List[bool]
    scene0: Dict[str, Any]
    cfg: Dict[str, Any]
    actions: List[_Action]
    action_num: int
    behaviors: List[dict]
    uid: Optional[str] = None
    rng_seed: Optional[int] = None
    passes: Optional[Any] = None


# -----------------------------------------------------------------------------
# public solver API
# -----------------------------------------------------------------------------


def solve_layout(
    labels: Sequence[Mapping[str, Any]],
    elements: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    config_profile: Union[str, Path, Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    return_viewpack: bool = False,
    deterministic_seed: Optional[int] = 42,
) -> SolveResult:
    """High level entry: ``from cartoweave import solve_layout``.

    This wrapper normalises raw inputs, merges configuration profiles and
    delegates the heavy lifting to the underlying solver.  It returns the
    label coordinates at the **end of each action** instead of every
    optimisation iteration.
    """
    elems_n, elem_maps = _normalize_elements(elements)
    labels_n = _normalize_labels(labels, elem_maps)
    acts_n = _normalize_actions(actions)
    _sanity_check_labels(labels_n)

    cfg = _load_and_merge_config(config_profile)
    if overrides:
        cfg = _deep_merge_dicts(cfg, overrides)

    if deterministic_seed is not None:
        _seed_everything(deterministic_seed)

    scene = {
        "frame_size": _infer_frame_size(labels_n),
        "labels": labels_n,
        "points": elems_n.get("points", []),
        "lines": elems_n.get("polylines", []),
        "areas": elems_n.get("polygons", []),
    }
    plan = {"actions": acts_n}

    sp = _build_pack_from_data(scene=scene, plan=plan, cfg=cfg)
    vp = _run_solver(sp)

    N = len(labels_n)
    coords = _extract_coords_from_viewpack(vp, N)
    return SolveResult(
        coords=coords,
        viewpack=(vp if return_viewpack else None),
        meta={"N": N, "T": int(coords.shape[0])},
    )


# ---------- 归一化 / 校验 ----------


def _normalize_labels(
    labels: Sequence[Mapping[str, Any]],
    elem_maps: Optional[Dict[str, Dict[Any, int]]] = None,
) -> List[Dict[str, Any]]:
    """Normalise label dictionaries and resolve element references."""
    elem_maps = elem_maps or {}
    out: List[Dict[str, Any]] = []
    for i, lab in enumerate(labels):
        d = dict(lab)
        lid = d.pop("label_id", d.pop("id", i))
        d["id"] = lid
        if "xy0" not in d and "xy" in d:
            d["xy0"] = d.pop("xy")
        if "WH" not in d:
            raise ValueError(f"Label {lid} missing WH")
        if "mode" not in d:
            d["mode"] = "rect"
        if d["mode"] == "circle" and isinstance(d["WH"], (list, tuple)) and len(d["WH"]) == 1:
            r = float(d["WH"][0])
            d["WH"] = [r, r]

        anch = d.get("anchor")
        if isinstance(anch, Mapping):
            k = anch.get("kind") or anch.get("target")
            idx = anch.get("index")
            if idx is None:
                elem_id = anch.get("element_id", anch.get("id"))
                idx = elem_maps.get(str(k), {}).get(elem_id)
                if idx is None:
                    raise ValueError(
                        f"Label {lid} references unknown {k} id {elem_id}"
                    )
            d["anchor"] = {"kind": k, "index": idx}
        if "xy0" not in d:
            # delay defaulting to anchor position until `_build_pack_from_data`
            if "anchor" not in d:
                d["xy0"] = [0.0, 0.0]

        out.append(d)
    return out


def _normalize_elements(
    elements: Mapping[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Dict[Any, int]]]:
    """Normalise scene elements and build id→index maps."""
    e = {"points": [], "polylines": [], "polygons": []}
    maps: Dict[str, Dict[Any, int]] = {"point": {}, "line": {}, "area": {}}
    if not elements:
        return e, maps

    # Points -----------------------------------------------------------------
    pts = elements.get("points", []) or []
    for i, item in enumerate(pts):
        if isinstance(item, Mapping):
            xy = item.get("xy") or item.get("xy0") or [0.0, 0.0]
            pid = item.get("element_id", item.get("id", i))
        else:
            xy = item
            pid = i
        e["points"].append(copy.deepcopy(xy))
        maps["point"][pid] = len(e["points"]) - 1

    # Polylines ---------------------------------------------------------------
    lines = elements.get("polylines", []) or []
    for i, item in enumerate(lines):
        if isinstance(item, Mapping):
            poly = item.get("polyline") or item.get("xy") or []
            lid = item.get("element_id", item.get("id", i))
        else:
            poly = item
            lid = i
        e["polylines"].append(copy.deepcopy(poly))
        maps["line"][lid] = len(e["polylines"]) - 1

    # Polygons ----------------------------------------------------------------
    polys = elements.get("polygons", []) or []
    for i, item in enumerate(polys):
        if isinstance(item, Mapping):
            poly = item.get("polygon") or item.get("xy") or []
            aid = item.get("element_id", item.get("id", i))
        else:
            poly = item
            aid = i
        e["polygons"].append(copy.deepcopy(poly))
        maps["area"][aid] = len(e["polygons"]) - 1

    return e, maps


def _normalize_actions(actions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if actions is None:
        return []
    out: List[Dict[str, Any]] = []
    for a in actions:
        d = dict(a)
        if "t" not in d:
            raise ValueError(f"Action missing 't': {a}")
        d.setdefault("op", "activate")
        if "label_ids" in d:
            d["ids"] = d.pop("label_ids")
        d.setdefault("ids", "all")
        if "action_id" in d:
            d["aid"] = d.pop("action_id")
        d.setdefault("payload", {})
        out.append(d)
    out.sort(key=lambda x: int(x["t"]))
    return out


def _sanity_check_labels(labels: Sequence[Mapping[str, Any]]) -> None:
    for i, d in enumerate(labels):
        xy0 = np.asarray(d.get("xy0", [0.0, 0.0]), dtype=float)
        WH = np.asarray(d.get("WH"), dtype=float)
        if xy0.shape != (2,) or not np.isfinite(xy0).all():
            raise ValueError(f"Label {i} invalid xy0: {xy0}")
        if WH.shape != (2,) or not np.isfinite(WH).all() or (WH <= 0).any():
            raise ValueError(f"Label {i} invalid WH: {WH}")
        mode = d.get("mode", "rect")
        if mode not in ("rect", "disk", "circle"):
            raise ValueError(f"Label {i} invalid mode: {mode}")


# ---------- YAML 合并 ----------


def _load_and_merge_config(profile: Optional[Union[str, Path, Mapping[str, Any]]]) -> Dict[str, Any]:
    if profile is None:
        return {}
    if isinstance(profile, Mapping):
        return copy.deepcopy(dict(profile))

    p = Path(profile)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if p.is_dir():
        cfg: Dict[str, Any] = {}
        for name in ["compute.public.yaml", "compute.tuning.yaml", "compute.internals.yaml"]:
            fp = p / name
            if fp.exists():
                with open(fp, "r", encoding="utf-8") as f:
                    cfg = _deep_merge_dicts(cfg, yaml.safe_load(f) or {})
        return cfg

    # handle "configs/compute" style pseudo-directory
    parent, base = p.parent, p.name
    cfg: Dict[str, Any] = {}
    found = False
    for name in [f"{base}.public.yaml", f"{base}.tuning.yaml", f"{base}.internals.yaml"]:
        fp = parent / name
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                cfg = _deep_merge_dicts(cfg, yaml.safe_load(f) or {})
                found = True
    if found:
        return cfg

    raise FileNotFoundError(f"Config profile not found: {profile}")


def _deep_merge_dicts(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------- 结果抽取 ----------


def _extract_coords_from_viewpack(viewpack: Any, N: int) -> np.ndarray:
    """Return coordinates only at the end of each action."""
    frames = getattr(viewpack, "frames", None)
    if not frames:
        coords = getattr(viewpack, "coords", None) or getattr(viewpack, "positions", None)
        if coords is None:
            raise ValueError("ViewPack missing frames/coords")
        arr = np.asarray(coords, float)
        assert arr.ndim == 3 and arr.shape[1:] == (N, 2), f"coords shape bad: {arr.shape}"
        return arr

    def _frame_xy(fr):
        P = getattr(fr, "P", None)
        if P is None:
            P = getattr(fr, "labels_xy", None)
        if P is None:
            P = getattr(fr, "xy", None)
        if P is None:
            raise ValueError("VPFrame missing label coordinates (P/labels_xy/xy)")
        P = np.asarray(P, dtype=float)
        if P.shape != (N, 2):
            if P.ndim == 2 and P.shape[1] == 2:
                Q = np.full((N, 2), np.nan, dtype=float)
                m = min(N, P.shape[0])
                Q[:m, :] = P[:m, :]
                P = Q
            else:
                raise AssertionError(
                    f"Frame P shape mismatch: got {P.shape}, expect {(N,2)}"
                )
        return P

    series: List[np.ndarray] = []
    passes = getattr(viewpack, "passes", None)
    if passes:
        for p in passes:
            fr = frames[p.t_end - 1]
            series.append(_frame_xy(fr))
    else:
        for fr in frames:
            series.append(_frame_xy(fr))
    return np.stack(series, axis=0)


# ---------- 杂项 ----------


def _seed_everything(seed: int) -> None:
    import os, random
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _infer_frame_size(
    labels: Sequence[Mapping[str, Any]],
    elements: Optional[Mapping[str, Any]] = None,
) -> tuple[float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for l in labels:
        xy0 = l.get("xy0")
        WH = l.get("WH")
        if xy0 is not None and WH is not None:
            xs.append(float(xy0[0]) + float(WH[0]))
            ys.append(float(xy0[1]) + float(WH[1]))
    if elements:
        for p in elements.get("points", []) or []:
            if isinstance(p, Mapping):
                xy = p.get("xy") or p.get("xy0") or [0.0, 0.0]
            else:
                xy = p
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
        for line in elements.get("polylines", []) or []:
            poly = line.get("polyline") if isinstance(line, Mapping) else line
            for x, y in poly:
                xs.append(float(x))
                ys.append(float(y))
        for poly in elements.get("polygons", []) or []:
            poly_coords = poly.get("polygon") if isinstance(poly, Mapping) else poly
            for x, y in poly_coords:
                xs.append(float(x))
                ys.append(float(y))
    W = (max(xs) if xs else 0.0) + 100.0
    H = (max(ys) if ys else 0.0) + 100.0
    return (W, H)


def _build_pack_from_data(scene: Dict[str, Any], plan: Dict[str, Any], cfg: Dict[str, Any]) -> _Pack:
    labels_raw = scene.get("labels", []) or []
    labels: List[_Label] = []
    P0: List[Sequence[Number]] = []
    for i, d in enumerate(labels_raw):
        anch = d.get("anchor") or {}
        if "kind" in anch and "index" in anch:
            k = anch.get("kind")
            idx = int(anch.get("index"))
            ax, ay, _meta = _anchor_xy(
                k, idx, scene, scene.get("frame_size", (0.0, 0.0)), with_meta=True
            )
            xy0 = tuple(map(float, d.get("xy0", (ax, ay))))
            anchor_info = {"kind": k, "index": idx, "xy": (ax, ay)}
        else:
            xy0 = tuple(map(float, d.get("xy0", (0.0, 0.0))))
            anchor_info = {"xy": xy0}
        lbl = _Label(
            id=int(d.get("id", i)),
            WH=tuple(map(float, d["WH"])),
            anchor=anchor_info,
            meta={"mode": d.get("mode", "rect")},
        )
        labels.append(lbl)
        P0.append(xy0)

    acts_in = plan.get("actions", []) or []
    actions: List[_Action] = []
    total_ids = sum(
        len(range(len(labels)) if a.get("ids", "all") == "all" else a.get("ids", []))
        for a in acts_in
    ) or len(labels)
    if total_ids == 0:
        total_ids = len(labels)
    t_counter = 1
    for a in acts_in:
        ids = list(range(len(labels))) if a.get("ids") == "all" else list(a.get("ids", []))
        op = a.get("op", "activate")
        payload = a.get("payload", {})
        aid = a.get("aid")
        for lid in ids:
            t = t_counter / (total_ids + 1)
            typ = (
                "appear"
                if op == "activate"
                else "disappear" if op == "deactivate" else "mutate"
            )
            actions.append(
                _Action(
                    label_id=int(lid),
                    t=float(t),
                    type=typ,
                    WH_to=tuple(payload.get("WH", labels[lid]["WH"])) if op != "deactivate" else payload.get("WH"),
                    kind_to=payload.get("mode"),
                    action_id=int(aid) if aid is not None else None,
                )
            )
            t_counter += 1
    if not actions:
        for i, lbl in enumerate(labels):
            t = (i + 1) / (len(labels) + 1)
            actions.append(_Action(label_id=i, t=t, type="appear", WH_to=tuple(lbl["WH"])))

    active0 = _derive_active0_from_actions(len(labels), actions).tolist()

    pack = _Pack(
        L=len(labels),
        P0=P0,
        labels0=labels,
        active0=active0,
        scene0=scene,
        cfg=cfg,
        actions=actions,
        action_num=len(actions),
        behaviors=[],
        uid=None,
        rng_seed=None,
    )
    return pack


def _derive_active0_from_actions(L, actions, t0=0.0, eps=1e-9):
    active0 = np.zeros((L,), dtype=bool)
    first: Dict[int, Any] = {}
    for a in actions:
        i = int(getattr(a, "label_id", getattr(a, "id", 0)))
        if i not in first or getattr(a, "t", 0.0) < getattr(first[i], "t", 0.0):
            first[i] = a
    for i in range(L):
        a = first.get(i)
        if a is None:
            active0[i] = True
        else:
            t = float(getattr(a, "t", 0.0))
            typ = str(getattr(a, "type", "appear"))
            if typ == "appear":
                active0[i] = t <= t0 + eps
            else:
                active0[i] = True
    return active0


__all__ = ["solve_layout", "SolveResult"]
