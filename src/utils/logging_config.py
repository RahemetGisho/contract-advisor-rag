"""Centralized logging configuration -- import get_logger(__name__)
instead of calling logging.basicConfig() ad hoc in multiple modules."""

import logging
import sys

_CONFIGURED = False


def _configure_root(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    if level is None:
        try:
            from src.utils.config import load_config

            level = load_config()["logging"]["level"]
        except Exception:
            level = "INFO"
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    _configure_root(level)
    return logging.getLogger(name)
