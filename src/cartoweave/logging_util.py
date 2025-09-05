# -*- coding: utf-8 -*-
from __future__ import annotations
import json, logging, os
from typing import Any, Dict


def get_logger(name: str, cfg: Dict[str, Any] | None = None) -> logging.Logger:
    level = "INFO"
    fmt = "text"
    if cfg:
        lg = cfg.get("compute", {}).get("logging", {})
        level = lg.get("level", level)
        fmt = lg.get("format", fmt)
    level = os.getenv("CARTOWEAVE_LOG_LEVEL", level)
    fmt = os.getenv("CARTOWEAVE_LOG_FORMAT", fmt)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        h = logging.StreamHandler(stream=os.sys.stdout)
        if fmt == "json":
            class _JSONF(logging.Formatter):
                def format(self, record):
                    base = {"level": record.levelname, "name": record.name}
                    extra = getattr(record, "extra", None)
                    if isinstance(extra, dict):
                        base.update(extra)
                    return json.dumps(base, ensure_ascii=False)
            h.setFormatter(_JSONF())
        else:
            h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(h)
    return logger
