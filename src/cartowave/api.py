from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import copy
import yaml
import numpy as np

# === 根据工程实际路径修正下方导入 ===
from cartoweave.contracts.solvepack import SolvePack
from cartoweave.compute.solve import solve as _run_solver

Number = Union[int, float]


@dataclass
class SolveResult:
    coords: np.ndarray            # (T, N, 2)
    viewpack: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None


def solve(
    labels: Sequence[Mapping[str, Any]],
    elements: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    config_profile: Union[str, Path, Mapping[str, Any]],
    overrides: Optional[Mapping[str, Any]] = None,
    return_viewpack: bool = False,
    deterministic_seed: Optional[int] = 42,
) -> SolveResult:
    """
    对外推荐入口：from cartowave import solve
    - 归一化入参 → 合并 YAML → 构造 SolvePack → 调 solver → 抽轨迹
    """
    labels_n = _normalize_labels(labels)
    elems_n  = _normalize_elements(elements)
    acts_n   = _normalize_actions(actions)
    _sanity_check_labels(labels_n)

    cfg = _load_and_merge_config(config_profile)
    if overrides:
        cfg = _deep_merge_dicts(cfg, overrides)

    if deterministic_seed is not None:
        _seed_everything(deterministic_seed)

    N = len(labels_n)
    scene = {
        "frame_size": _infer_frame_size(labels_n),
        "points":    elems_n.get("points", []),
        "lines":     elems_n.get("polylines", []),
        "areas":     elems_n.get("polygons", []),
    }
    plan = {"actions": acts_n}

    sp = _build_solvepack(labels_n, scene, plan, cfg)
    vp = _run_solver(sp)

    coords = _extract_coords_from_viewpack(vp, N)
    return SolveResult(coords=coords, viewpack=(vp if return_viewpack else None),
                       meta={"N": N, "T": int(coords.shape[0])})


# ---------- 归一化 / 校验 ----------

def _normalize_labels(labels: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, lab in enumerate(labels):
        d = dict(lab)
        d.setdefault("id", i)
        if "xy0" not in d and "xy" in d:
            d["xy0"] = d.pop("xy")
        if "WH" not in d:
            raise ValueError(f"Label {d.get('id', i)} missing WH")
        if "mode" not in d:
            d["mode"] = "rect"
        if d["mode"] == "circle" and isinstance(d["WH"], (list, tuple)) and len(d["WH"]) == 1:
            r = float(d["WH"][0])
            d["WH"] = [r, r]
        out.append(d)
    return out


def _normalize_elements(elements: Mapping[str, Any]) -> Dict[str, Any]:
    e = {"points": [], "polylines": [], "polygons": []}
    if not elements:
        return e
    for k in e.keys():
        if k in elements and elements[k]:
            e[k] = copy.deepcopy(elements[k])
    return e


def _normalize_actions(actions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if actions is None:
        return []
    out: List[Dict[str, Any]] = []
    for a in actions:
        d = dict(a)
        if "t" not in d:
            raise ValueError(f"Action missing 't': {a}")
        d.setdefault("op", "activate")
        d.setdefault("target", "label")
        d.setdefault("ids", "all")
        d.setdefault("payload", {})
        out.append(d)
    out.sort(key=lambda x: int(x["t"]))
    return out


def _sanity_check_labels(labels: Sequence[Mapping[str, Any]]) -> None:
    for i, d in enumerate(labels):
        xy0 = np.asarray(d.get("xy0", [0.0, 0.0]), dtype=float)
        WH  = np.asarray(d.get("WH"), dtype=float)
        if xy0.shape != (2,) or not np.isfinite(xy0).all():
            raise ValueError(f"Label {i} invalid xy0: {xy0}")
        if WH.shape != (2,) or not np.isfinite(WH).all() or (WH <= 0).any():
            raise ValueError(f"Label {i} invalid WH: {WH}")
        mode = d.get("mode", "rect")
        if mode not in ("rect", "disk", "circle"):
            raise ValueError(f"Label {i} invalid mode: {mode}")


# ---------- SolvePack 构造 ----------

def _build_solvepack(labels: List[Dict[str, Any]], scene: Dict[str, Any], plan: Dict[str, Any], cfg: Dict[str, Any]) -> SolvePack:
    N = len(labels)
    action_records = _build_action_records(N, plan.get("actions", []))
    active0 = _derive_active0_from_actions(N, action_records)
    P0 = [list(map(float, lab.get("xy0", (0.0, 0.0)))) for lab in labels]
    label_objs = [
        {
            "id": int(lab["id"]),
            "kind": "point",
            "WH": list(map(float, lab["WH"])),
            "anchor": {"target": "free", "mode": "xy", "xy": list(map(float, lab.get("xy0", (0.0, 0.0))))},
            "mode": lab.get("mode"),
        }
        for lab in labels
    ]
    sp = SolvePack(
        L=N,
        P0=P0,
        labels0=label_objs,
        active0=active0,
        scene0=scene,
        cfg={"compute": cfg},
        actions=action_records,
        action_num=len(action_records),
        behaviors=[],
    )
    return sp


def _build_action_records(N: int, actions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not actions:
        ids = list(range(N))
        dt = 1.0 / (N + 1)
        return [{"t": (i + 1) * dt, "id": i, "type": "appear"} for i in ids]
    recs: List[Dict[str, Any]] = []
    total_ids = sum(len(range(N)) if act.get("ids") in ("all", "*") else len(act.get("ids", [])) for act in actions)
    total_ids = total_ids or N
    dt = 1.0 / (total_ids + 1)
    t = dt
    for act in actions:
        ids = list(range(N)) if act.get("ids") in ("all", "*") else list(act.get("ids", []))
        for i in ids:
            recs.append({"t": t, "id": int(i), "type": "appear"})
            t += dt
    return recs


def _derive_active0_from_actions(L: int, actions: Sequence[Mapping[str, Any]], t0: float = 0.0, eps: float = 1e-9) -> List[bool]:
    active0 = np.zeros((L,), dtype=bool)
    first: Dict[int, Dict[str, Any]] = {}
    for a in actions:
        i = int(a.get("id", 0))
        if i not in first or float(a.get("t", 0.0)) < float(first[i].get("t", 0.0)):
            first[i] = a
    for i in range(L):
        a = first.get(i)
        if a is None:
            active0[i] = True
        else:
            t = float(a.get("t", 0.0))
            typ = str(a.get("type", "appear"))
            if typ == "appear":
                active0[i] = t <= t0 + eps
            else:
                active0[i] = True
    return active0.tolist()


def _infer_frame_size(labels: Sequence[Mapping[str, Any]]) -> List[float]:
    xs = [float(l["xy0"][0]) for l in labels]
    ys = [float(l["xy0"][1]) for l in labels]
    max_x = max(xs) if xs else 0.0
    max_y = max(ys) if ys else 0.0
    return [max(256.0, max_x * 2 + 256.0), max(256.0, max_y * 2 + 256.0)]


# ---------- YAML 合并 ----------

def _load_and_merge_config(profile: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(profile, Mapping):
        return copy.deepcopy(dict(profile))

    p = Path(profile)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if p.is_dir():
        cfg: Dict[str, Any] = {}
        name_groups = [
            ["compute_public.yaml", "compute.public.yaml"],
            ["compute_tuning.yaml", "compute.tuning.yaml"],
            ["compute_internal.yaml", "compute.internals.yaml"],
        ]
        for names in name_groups:
            for name in names:
                fp = p / name
                if fp.exists():
                    with open(fp, "r", encoding="utf-8") as f:
                        cfg = _deep_merge_dicts(cfg, yaml.safe_load(f) or {})
                    break
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
    frames = getattr(viewpack, "frames", None)
    if not frames:
        coords = getattr(viewpack, "coords", None)
        if coords is None:
            coords = getattr(viewpack, "positions", None)
        if coords is None:
            raise ValueError("ViewPack missing frames/coords")
        arr = np.asarray(coords, float)
        assert arr.ndim == 3 and arr.shape[1:] == (N, 2), f"coords shape bad: {arr.shape}"
        return arr

    series: List[np.ndarray] = []
    for fr in frames:
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
                raise AssertionError(f"Frame P shape mismatch: got {P.shape}, expect {(N,2)}")
        series.append(P)
    return np.stack(series, axis=0)


# ---------- 杂项 ----------

def _seed_everything(seed: int) -> None:
    import os, random
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
