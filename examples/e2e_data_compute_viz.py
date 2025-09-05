# Optional YAML config; keep None to use built-in defaults.
YAML_PATH = None  # e.g., "examples/configs/accessible/basic.yaml"

# Reproducible random seed for scene + script
SEED = 123

# When a script is very long, you may cap the number of steps (None = all)
MAX_STEPS = None      # or set to a small int like 12 to keep runtime short

# Per-step LBFGS iterations (applied to every script step/stage)
PER_STEP_ITERS = 6

import os
import numpy as np
import sys

# Allow running directly from repository root without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cartoweave.data.random import get_scene   # MUST use this to obtain scene + scene_script
from cartoweave.compute.pack import SolvePack
from cartoweave.compute.run import solve

from cartoweave.viz.build_viz_payload import build_viz_payload
from cartoweave.viz.view import interactive_view
from cartoweave.viz.defaults import VIZ_DEFAULTS
from cartoweave.viz.backend import use_compatible_backend

try:
    import yaml  # optional; if missing, fall back to built-in config
except Exception:
    yaml = None


def load_scene_with_script(seed: int = SEED):
    """
    Use get_scene(use_random=True, with_scene_script=True, gen_cfg=None, seed=seed, ...)
    to produce a dict with:
      - frame_size: (W,H)
      - points: (Np,2) ndarray (or list convertible to ndarray)
      - lines:  List[(Ni,2) ndarray]
      - areas:  List[{"polygon": (Ni,2) ndarray}] or List[(Ni,2) ndarray]
      - labels_init: (L,2) ndarray
      - WH: (L,2) ndarray  (per-label w/h)
      - labels: List[dict] with 'anchor_kind' in {"none","point","line","area"} and 'anchor_index'
      - scene_script: List[{"op": "appear"|"change"|"hide", "id": "pN|lN|aN", "mode"?: str, "step_id": int}]
    """
    data = get_scene(use_random=True, cache_path=None, with_scene_script=True,
                     gen_cfg=None, seed=seed)
    return data


def ensure_scene_fields(scene):
    assert isinstance(scene.get("labels_init"), np.ndarray)
    assert scene["labels_init"].ndim == 2 and scene["labels_init"].shape[1] == 2
    L = scene["labels_init"].shape[0]

    assert "WH" in scene and isinstance(scene["WH"], np.ndarray), "scene['WH'] must exist"
    assert scene["WH"].shape == (L, 2), f"scene['WH'] must be (L,2), got {scene['WH'].shape}"

    assert "labels" in scene and len(scene["labels"]) == L
    assert "frame_size" in scene and len(scene["frame_size"]) == 2

    if isinstance(scene.get("points"), list):
        scene["points"] = np.asarray(scene["points"], float)
    scene["lines"]  = [np.asarray(Li, float) for Li in scene.get("lines", [])]
    # 'areas' may be a list of dicts with 'polygon' ndarray
    return scene


def anchor_eid_for_label(lb) -> str | None:
    k = lb.get("anchor_kind")
    idx = lb.get("anchor_index", 0)
    if k == "point": return f"p{int(idx)}"
    if k == "line":  return f"l{int(idx)}"
    if k == "area":  return f"a{int(idx)}"
    return None  # 'none' etc.


def stages_from_scene_script(scene, scene_script, per_step_iters: int, max_steps=None):
    L = scene["labels_init"].shape[0]
    labels = scene["labels"]

    active_elems = set()
    elem_mode = {}   # id -> mode (updated on "appear"/"change")
    stages = []
    step_meta = []   # carry through to viz

    script = scene_script if max_steps is None else scene_script[:int(max_steps)]

    for s_idx, act in enumerate(script):
        op = act.get("op")
        eid = str(act.get("id"))
        mode = act.get("mode")

        if op == "appear":
            active_elems.add(eid)
            if mode is not None:
                elem_mode[eid] = str(mode)
        elif op == "change":
            if mode is not None:
                elem_mode[eid] = str(mode)
        elif op == "hide":
            active_elems.discard(eid)

        # Build mask
        mask = np.zeros(L, dtype=bool)
        for i, lb in enumerate(labels):
            eid_i = anchor_eid_for_label(lb)
            if eid_i is None:
                mask[i] = True  # 'none' -> always active
            elif eid_i in active_elems:
                mask[i] = True

        # Visible layers for viz
        vis_layers = sorted({ {"p":"points","l":"lines","a":"areas"}[e[0]] for e in active_elems }) if active_elems else []

        stages.append({
            "solver": "lbfgs",
            "params": {"lbfgs_maxiter": int(per_step_iters), "lbfgs_pgtol": 1e-6},
            "mask": mask,
            "name": f"script_{s_idx:03d}_{op}_{eid}",
        })
        step_meta.append({
            "id": s_idx,
            "op": op,
            "element_id": eid,
            "mode": elem_mode.get(eid),
            "visible_layers": vis_layers,
            "active_label_ids": np.flatnonzero(mask).tolist(),
        })
    return stages, step_meta


def load_cfg_yaml(path: str | None, frame_size):
    W, H = frame_size
    if path and yaml:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "compute": {
            "eps": {"numeric": 1e-12},
            "weights": {
                "anchor.spring": 1.0,
                "pl.rect": 0.3, "ln.rect": 0.4, "ll.rect": 0.6,
                "area.embed": 0.2, "boundary.wall": 0.2,
                "focus.attract": 0.0,
            },
            "passes": {
                "capture":   {"every": 1, "final_always": True},
                "grad_clip": {"max_norm": None, "max_abs": None},
                "step_limit":{"max_step_norm": None},
            },
        },
        # term knobs (kept at top-level for back-compat)
        "boundary.k.in": 1.0,
        "boundary.wall_power": 2.0,
        "pl.k.inside": 1.0, "pl.k.repulse": 1.0,
        "ln.k.inside": 1.0, "ln.k.repulse": 1.0,
        "ll.k.inside": 1.0, "ll.k.repulse": 1.0,
        "focus.center": [W*0.5, H*0.5],
        "focus.k.attract": 0.0,
    }


def run_solve(scene, cfg, stages):
    L = scene["labels_init"].shape[0]
    sp = SolvePack(
        L=L,
        P0=scene["labels_init"],
        active_mask0=np.ones(L, dtype=bool),
        scene=scene,
        cfg=cfg,
        stages=stages,
        passes=["schedule", "capture"],
    )
    return solve(sp)


def _scene_steps_from_frames(frames):
    groups, steps = {}, []
    for t, f in enumerate(frames):
        groups.setdefault(f.stage, []).append(t)
    for s, idxs in sorted(groups.items()):
        steps.append({"id": s, "name": f"stage_{s}", "rec_start": idxs[0], "rec_end": idxs[-1]+1})
    return steps


def to_viz_payload(view, scene, step_meta, scene_script):
    records = [
        {"P": f.P, "G": f.G, "E": float(f.E),
         "comps": f.comps or {}, "mask": f.mask, "meta": f.meta or {}}
        for f in view.frames
    ]
    scene_steps = _scene_steps_from_frames(view.frames)

    # Attach meta (op/element_id/mode/visible_layers/active_label_ids) per step index
    for s in range(min(len(scene_steps), len(step_meta))):
        scene_steps[s].update(step_meta[s])

    # Build actions: mark 'appear/change/hide' at each step start
    actions = []
    for s, st in enumerate(scene_steps):
        t0 = st["rec_start"]
        m = step_meta[s] if s < len(step_meta) else {}
        actions.append({"t": t0, "kind": m.get("op","op"), "id": m.get("element_id"), "mode": m.get("mode")})

    info = {
        "history": {"records": records, "scene_steps": scene_steps},
        "scene": {
            "labels": scene.get("labels", []),
            "points": scene.get("points", []),
            "lines":  scene.get("lines",  []),
            "areas":  scene.get("areas",  []),
        },
        "actions": actions,
    }
    payload = build_viz_payload(info)
    if "actions" not in payload:
        payload["actions"] = actions
    return payload


def render_static_png(payload, scene, out_path="out/e2e_final.png"):
    use_compatible_backend()
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    W, H = scene["frame_size"]
    fr_last = payload["frames"][-1]
    P = np.asarray(fr_last["P"], float)
    WH = np.asarray(scene["WH"], float)

    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_title("CartoWeave result (final frame)")

    # background geometries
    for a in scene.get("areas", []):
        poly = a.get("polygon") if isinstance(a, dict) else np.asarray(a, float)
        ax.fill(poly[:,0], poly[:,1], alpha=0.08, linewidth=1, edgecolor="k")
    for Ls in scene.get("lines", []):
        arr = np.asarray(Ls, float); ax.plot(arr[:,0], arr[:,1], linewidth=1.2)
    pts = np.asarray(scene.get("points", []), float)
    if pts.size:
        ax.scatter(pts[:,0], pts[:,1], s=18)

    # labels as rectangles
    for i, p in enumerate(P):
        w, h = float(WH[i,0]), float(WH[i,1])
        rect = plt.Rectangle((p[0]-w/2, p[1]-h/2), w, h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(p[0], p[1], str(i), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def try_interactive(payload, scene):
    try:
        traj = np.stack([fr["P"] for fr in payload["frames"]], axis=0)
        labels = scene["labels"]
        rect_wh = np.asarray(scene["WH"], float)
        areas_arr = [a.get("polygon") if isinstance(a, dict) else np.asarray(a, float)
                     for a in scene.get("areas", [])]
        W, H = scene["frame_size"]
        interactive_view(
            traj, labels, rect_wh,
            scene.get("points", []), scene.get("lines", []), areas_arr,
            W, H,
            frames=payload["frames"],
            boundaries=payload["boundaries"],
            actions=payload.get("actions", []),
            viz=VIZ_DEFAULTS,
        )
    except Exception:
        pass


def main(yaml_path: str | None = YAML_PATH):
    data = load_scene_with_script(SEED)
    scene = ensure_scene_fields(data)
    script = list(scene.get("scene_script", []))
    if MAX_STEPS is not None:
        script = script[:int(MAX_STEPS)]

    cfg = load_cfg_yaml(yaml_path, scene["frame_size"])

    stages, step_meta = stages_from_scene_script(scene, script,
                                                 per_step_iters=PER_STEP_ITERS,
                                                 max_steps=None)

    view = run_solve(scene, cfg, stages)
    payload = to_viz_payload(view, scene, step_meta, script)

    try_interactive(payload, scene)
    #out_png = render_static_png(payload, scene, "out/e2e_final.png")
    #print(f"[OK] seed={SEED} steps={len(stages)} frames={len(payload['frames'])}, saved: {out_png}")


if __name__ == "__main__":
    main()