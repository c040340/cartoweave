"""Minimal e2e script building a pack and exiting before solving."""

from __future__ import annotations

import argparse, json

from cartoweave.config.loader import load_configs
from cartoweave.data.api import build_solvepack_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Profile YAML path")
    ap.add_argument("--overrides", type=str, default=None, help="JSON overrides")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=8, help="How many behaviors to print")
    args = ap.parse_args()

    overrides = json.loads(args.overrides) if args.overrides else None
    merged = load_configs(profile=args.config, overrides=overrides)

    pack = build_solvepack_from_config(merged, seed=args.seed)
    pack.validate()

    scene = pack.scene0
    behaviors = list(pack.cfg.get("behaviors", []))
    behavior_cfg = dict(pack.cfg.get("behavior", {}))
    N = int(pack.N)
    P = int(len(scene.points))
    L = int(len(scene.lines))
    A = int(len(scene.areas))
    G = P + L + A
    S = int(len(behaviors))

    # --- Summary ---
    print(f"[e2e] N={N} steps={S} seed={args.seed}")
    print(f"[e2e] scene points/lines/areas = {P}/{L}/{A}")
    if behavior_cfg:
        apol = behavior_cfg.get("anchor_policy", "round_robin")
        dwh = behavior_cfg.get("default_WH", {})
        print(f"[e2e] behavior: anchor_policy={apol}, default_WH={dwh}")

    # --- Prominent sanity warnings ---
    if S < G:
        print(f"⚠️  [sanity] steps ({S}) < points+lines+areas ({G}). "
              f"Some geometries may never get a first activation.")
    if N != G:
        print(f"⚠️  [sanity] label count N ({N}) != points+lines+areas ({G}). "
              f"This is allowed, but check your design intent.")

    # --- First-activation health check per label ---
    first_act_idx = [-1] * N
    first_act_mut_ok = [False] * N  # has kind+WH+(anchor if available)
    for sidx, b in enumerate(behaviors):
        ops = getattr(b, "ops", b.get("ops", {})) if isinstance(b, dict) else b.ops
        acts = ops.get("activate", [])
        muts = ops.get("mutate", [])
        # index mutate by id for quick lookup
        mut_by_id = {}
        for m in muts:
            try:
                mut_by_id.setdefault(int(m["id"]), []).append(m.get("set", {}))
            except Exception:
                pass
        for lid in acts:
            if 0 <= int(lid) < N and first_act_idx[int(lid)] == -1:
                first_act_idx[int(lid)] = sidx
                sets = mut_by_id.get(int(lid), [])
                # consider ok if we have WH and (kind or anchor.kind)
                ok = False
                for sv in sets:
                    has_wh = isinstance(sv.get("WH"), (list, tuple)) and len(sv.get("WH", [])) == 2
                    k = sv.get("kind")
                    anc = sv.get("anchor", {})
                    ak = anc.get("kind") if isinstance(anc, dict) else None
                    if has_wh and (k in {"point", "line", "area"} or ak in {"point", "line", "area"}):
                        ok = True
                        break
                first_act_mut_ok[int(lid)] = ok

    missing_act = [i for i, sidx in enumerate(first_act_idx) if sidx == -1]
    weak_act   = [i for i, ok in enumerate(first_act_mut_ok) if not ok and first_act_idx[i] != -1]

    if missing_act:
        print(f"⚠️  [sanity] {len(missing_act)}/{N} labels NEVER activated. "
              f"Examples: {missing_act[:min(5,len(missing_act))]}")
    if weak_act:
        print(f"⚠️  [sanity] {len(weak_act)}/{N} labels first activation lacks kind/WH (and/or anchor). "
              f"Examples: {weak_act[:min(5,len(weak_act))]}")

    # --- Print a concise preview of behaviors ---
    show = max(0, min(args.show, S))
    if show:
        print(f"[e2e] behaviors preview (first {show}/{S}):")
        for sidx in range(show):
            b = behaviors[sidx]
            iters = getattr(b, "iters", b.get("iters", None)) if isinstance(b, dict) else b.iters
            solver = getattr(b, "solver", b.get("solver", None)) if isinstance(b, dict) else b.solver
            ops = getattr(b, "ops", b.get("ops", {})) if isinstance(b, dict) else b.ops
            act = ops.get("activate", [])
            deact = ops.get("deactivate", [])
            muts = ops.get("mutate", [])
            # compact mutate summary: only show first mutate with keys present
            msum = []
            if muts:
                m0 = muts[0]
                sid = m0.get("id", None)
                sset = m0.get("set", {})
                k = sset.get("kind", None)
                wh = sset.get("WH", None)
                anc = sset.get("anchor", None)
                if k is not None: msum.append(f"kind={k}")
                if wh is not None: msum.append(f"WH={wh}")
                if isinstance(anc, dict) and "kind" in anc and "index" in anc:
                    msum.append(f"anchor={anc['kind']}#{anc['index']}")
            msum_txt = ", ".join(msum) if msum else "-"
            print(f"  - step {sidx:02d}: iters={iters}, solver={solver}, "
                  f"activate={act}, deactivate={deact}, mutate_first=[{msum_txt}]")

    print("[e2e] pack ready (pre-solve)")


if __name__ == "__main__":
    main()
