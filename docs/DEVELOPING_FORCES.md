# Developing a New Force Term (compute-only)

Each force term lives in `src/cartoweave/compute/forces/<name>.py` and registers itself:

```py
from . import register
import numpy as np

@register("myterm.foo")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
  """
  Inputs:
    - scene: read-only view; do not mutate arrays in-place
    - P: (L,2) float array for current label positions
    - cfg: dict, read from cfg['compute'] (use helpers for eps/weights)
    - phase: "pre_anchor" or "anchor"
  Returns: (E, F, meta)
    - E: scalar energy (float)
    - F: (L,2) force; DO NOT zero inactive rows here (aggregator does it)
    - meta: small dict with diagnostics; avoid large payloads
  """
  # Example skeleton
  from ._common import get_eps, weight_of, ensure_vec2
  eps = get_eps(cfg)
  w   = weight_of("myterm.foo", cfg, 0.0)
  if w <= 0.0:
      return 0.0, np.zeros_like(P), {"disabled": True}

  # --- compute raw energy and raw force F_raw ---
  E_raw = 0.0
  F_raw = np.zeros_like(P)

  # Check shape and apply weight
  F = ensure_vec2(F_raw, P.shape[0])
  return float(E_raw * w), F * w, {"source": "compute.forces.myterm.foo"}
```

## Rules

- Keep all numerical guards via `get_eps(cfg)` (no hard-coded 1e-12).
- No per-term clipping; `GradClipPass` handles that globally.
- Aggregator zeroes inactive rows; don’t do it in the term.
- If your term has multiple keys (e.g., `myterm.foo`, `myterm.bar`), call `@register(...)` once per key or split into functions.

## Testing

Add a parity or sanity test in `tests/`, asserting shapes, finiteness, and (if applicable) comparison to a reference implementation.
