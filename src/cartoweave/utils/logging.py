"""Logging utilities for cartoweave.

This module centralises logging configuration so that all submodules use the
same logger instance.  By default the logger is silent (a ``NullHandler`` is
installed) and can be enabled or disabled globally via :func:`configure_logging`.
"""

import logging


# Global project-wide logger -------------------------------------------------
logger = logging.getLogger("cartoweave")
logger.addHandler(logging.NullHandler())


def configure_logging(enabled: bool = True, level: int = logging.INFO) -> None:
    """Configure the global ``cartoweave`` logger.

    Parameters
    ----------
    enabled:
        If ``True`` (default) a ``StreamHandler`` is installed.  If ``False``
        logging output is suppressed.
    level:
        Logging level used when enabling the handler.  Iterative solver messages
        are emitted at ``DEBUG`` level to avoid overwhelming the console unless
        explicitly requested.
    """

    # Remove any existing handlers so configuration calls are idempotent.
    logger.handlers.clear()

    if enabled:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    else:
        # Install a ``NullHandler`` and bump the log level so no messages leak.
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)


__all__ = ["logger", "configure_logging"]

