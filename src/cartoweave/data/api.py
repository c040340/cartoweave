"""Helpers to construct :class:`SolvePack` objects for tests and examples."""
from __future__ import annotations

from typing import Any

import numpy as np

from cartoweave.config.loader import load_data_defaults
from cartoweave.contracts.solvepack import SolvePack

from .generate import generate_scene
from cartoweave.data.io import load_snapshot
from .action_sequence import generate_action_sequence_strict
from .textblock import load_font

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
    if cfg.source == "load":
        load_cfg = cfg.load  # pydantic 已保证存在
        scene0, P0, active0, labels0, actions, action_num = load_snapshot(load_cfg.path)
        pack = SolvePack(
                L = len(labels0),
            P0 = np.asarray(P0, dtype=float).tolist(),
            labels0 = labels0,
            active0 = np.asarray(active0, dtype=bool).tolist(),
            scene0 = scene0,
            cfg = {"compute": compute_cfg or {}},
        )
        return pack
    elif cfg.source != "generate":  # pragma: no cover - safety
        raise ValueError(f"unknown data.source: {cfg.source}")

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

    S = cfg.action_num
    txt_cfg = gen.text
    font = load_font(txt_cfg.font.path, int(txt_cfg.font.size))
    len_min, len_max = map(int, txt_cfg.len_range)
    spacing = int(txt_cfg.line_spacing_px)
    padx = int(txt_cfg.padding_px.x)
    pady = int(txt_cfg.padding_px.y)
    resample = bool(gen.mutate.resample_text_on_size_mutate)
    actions = generate_action_sequence_strict(
        labels0,
        S,
        rng,
        font,
        (len_min, len_max),
        spacing,
        padx,
        pady,
        resample,
    )
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

