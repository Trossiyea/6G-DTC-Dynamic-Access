"""
Lightweight logging helpers to keep the library silent by default while
letting callers opt-in to structured logs.
"""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with a NullHandler attached so library code does not emit
    logs unless configured by the caller.
    """
    logger = logging.getLogger(name)
    # Avoid stacking multiple NullHandlers if called repeatedly
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def configure_logging(level: int = logging.INFO, fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s") -> None:
    """
    Optional convenience for CLIs/web apps to enable logging.
    """
    logging.basicConfig(level=level, format=fmt)
