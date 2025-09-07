"""Helpers to construct :class:`SolvePack` objects for tests and examples."""
from __future__ import annotations

from typing import Any

import numpy as np

from cartoweave.config.loader import load_data_defaults
from cartoweave.contracts.solvepack import SolvePack

from .generate import generate_scene
from .action_sequence import generate_action_sequence_strict

__all__ = ["make_solvepack_from_data_defaults"]


def make_solvepack_from_data_defaults(
    data_path: str = "configs/data.yaml",
    compute_cfg: dict[str, Any] | None = None,
) -> SolvePack:
    """Build a :class:`SolvePack` using ``data_path`` defaults.

    Parameters
    ----------
    data_path:
        Path to data configuration file.
    compute_cfg:
        Optional compute configuration injected into the SolvePack ``cfg``.
    """

    cfg = load_data_defaults(data_path)
    if cfg.source != "generate":  # pragma: no cover - non-default path
        raise NotImplementedError("loading scenes from file is not supported")

    gen = cfg.generate
    if gen is None:  # pragma: no cover - config validation ensures non-None
        raise ValueError("generate config required")
    rng = np.random.default_rng(gen.seed or 0)
    p0, labels0, active0, scene0 = generate_scene(gen, rng)

    pack = SolvePack(
        L=len(labels0),
        P0=[(float(x), float(y)) for x, y in p0],
        labels0=labels0,
        active0=active0.tolist(),
        scene0=scene0,
        cfg={"compute": compute_cfg or {}},
        actions=[],
        action_num=None,
        behaviors=[],
    )

    behaviors = getattr(cfg, "behaviors", None)
    if behaviors:
        pack.behaviors = list(behaviors)
        pack.action_num = len(behaviors)
    else:
        S = cfg.action_num
        actions = generate_action_sequence_strict(len(labels0), S, rng)
        pack.actions = actions
        pack.action_num = S

    gen_cfg = gen
    save_path = getattr(gen_cfg, "save_path", None) or "./snapshots/default_pack.json"
    from cartoweave.data.io import save_snapshot
    try:
        save_snapshot(pack, save_path, fmt="json")
    except Exception as e:  # pragma: no cover - fail fast
        raise RuntimeError(f"Autosave failed to '{save_path}': {e}")

    return pack

