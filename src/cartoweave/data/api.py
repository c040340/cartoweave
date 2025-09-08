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


def _derive_active0_from_actions(L, actions, t0=0.0, eps=1e-9):
    import numpy as np

    active0 = np.zeros((L,), dtype=bool)
    first = {}
    for a in actions:
        i = int(getattr(a, "id", getattr(a, "label_id", 0)))
        if i not in first or getattr(a, "t", 0.0) < getattr(first[i], "t", 0.0):
            first[i] = a
    for i in range(L):
        a = first.get(i, None)
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
        scene0, P0, _active0, labels0, actions, action_num = load_snapshot(load_cfg.path)
        active0 = _derive_active0_from_actions(len(labels0), actions)
        pack = SolvePack(
            L=len(labels0),
            P0=np.asarray(P0, dtype=float).tolist(),
            labels0=labels0,
            active0=active0.tolist(),
            scene0=scene0,
            cfg={"compute": compute_cfg or {}},
            actions=actions,
            action_num=action_num,
            behaviors=[],
        )
        return pack
    elif cfg.source != "generate":  # pragma: no cover - safety
        raise ValueError(f"unknown data.source: {cfg.source}")

    gen = cfg.generate
    if gen is None:  # pragma: no cover - config validation ensures non-None
        raise ValueError("generate config required")
    rng = np.random.default_rng(gen.seed or 0)
    p0, labels0, scene0 = generate_scene(gen, rng)

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
    active0 = _derive_active0_from_actions(len(labels0), actions)

    pack = SolvePack(
        L=len(labels0),
        P0=[(float(x), float(y)) for x, y in p0],
        labels0=labels0,
        active0=active0.tolist(),
        scene0=scene0,
        cfg={"compute": compute_cfg or {}},
        actions=actions,
        action_num=S,
        behaviors=[],
    )

    gen_cfg = gen
    save_path = getattr(gen_cfg, "save_path", None) or "./snapshots/default_pack.json"
    from cartoweave.data.io import save_snapshot
    try:
        save_snapshot(pack, save_path, fmt="json")
    except Exception as e:  # pragma: no cover - fail fast
        raise RuntimeError(f"Autosave failed to '{save_path}': {e}")

    return pack

