"""Logging setup (stdlib + optional wandb).

A minimal rank-aware logger plus an optional Weights & Biases hook (disabled by default;
training and evaluation run fine without it). The shared library logger is named ``rdm``.
"""
import logging
import sys

LOGGER_NAME = "rdm"


def setup_logging(level: int = logging.INFO, rank: int = 0) -> logging.Logger:
    """Configure the ``rdm`` logger to stdout; only rank 0 logs at ``level`` (others WARNING)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                            datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level if rank == 0 else logging.WARNING)
    logger.propagate = False
    return logger


def setup_wandb(project: str, name: str | None = None, config: dict | None = None,
                enabled: bool = False, **kwargs):
    """Initialize wandb if ``enabled`` and importable; otherwise return ``None`` (a no-op)."""
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        logging.getLogger(LOGGER_NAME).warning("wandb not installed; logging disabled")
        return None
    return wandb.init(project=project, name=name, config=config, **kwargs)
