# Compute module quick reference

- **Solve**: `ViewPack = solve(SolvePack)`
- **Aggregator**: `compute.eval.energy_and_grad_full` (compute-only)
- **Add force term**: create `compute/forces/<name>.py` and `@register("<key>")`
- **Passes**: `schedule`, `capture`, `nan_guard`, `grad_clip`, `step_limit`
- **Schedule example**:
  ```python
  sp.cfg.setdefault("compute", {}).setdefault("public", {}).setdefault("forces", {})[
      "anchor.spring"
  ] = {"enable": True, "k": 2.0}
  sp.passes = ["schedule", {"name": "capture", "args": {"every": 1}}]
  sp.schedule = [{"solver": "lbfgs", "iters": 5}, {"solver": "semi", "iters": 5}]
  vp = solve(sp)
  ```
- **Invariants**: arrays are `(L,2)`, inactive rows are zero, `G ≈ -Σcomps`
- **Testing**:
  - `pytest -q`
  - Verify aggregation:
    ```python
    E,G,comps,_ = energy_and_grad_full(P, scene, mask, cfg)
    assert np.allclose(G, -sum(comps.values()), atol=1e-6)
    ```
