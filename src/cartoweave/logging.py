from __future__ import annotations
import logging
from typing import Optional, Mapping, Any

_LEVEL_MAP = {
    "none": logging.WARNING,  # 仅打印 WARNING/ERROR/CRITICAL
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

def _normalize(level: Optional[str]) -> int:
    if not level:
        return logging.WARNING
    return _LEVEL_MAP.get(str(level).strip().lower(), logging.WARNING)

def level_from_cfg(cfg: Optional[Mapping[str, Any]]) -> int:
    """
    仅从 compute.solver.public.log_level 读取；若不存在或为空，则返回 WARNING。
    注意：不读取其它 YAML；本项目只在 compute.public.yaml 中配置。
    """
    if not isinstance(cfg, Mapping):
        return logging.WARNING
    try:
        comp = cfg.get("compute", {}) or {}
        solver_pub = (comp.get("solver", {}) or {}).get("public", {}) or {}
        return _normalize(solver_pub.get("log_level", "none"))
    except Exception:
        return logging.WARNING

def init_logging(level: int | str | None = None) -> None:
    """
    初始化根 logger 与 cartoweave 命名空间 logger；避免重复 handler。
    多次调用允许覆盖等级，但不重复添加 handler。
    """
    lvl = _normalize(level) if isinstance(level, str) or level is None else int(level)

    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(h)
    root.setLevel(lvl)

    logging.getLogger("cartoweave").setLevel(lvl)

def init_logging_from_cfg(cfg: Optional[Mapping[str, Any]]) -> None:
    init_logging(level_from_cfg(cfg))
