"""Logging helpers for the backend application."""

import logging
from pathlib import Path

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with a console handler."""
    logger = logging.getLogger(f"eoa.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def get_session_logger(session_id: str) -> logging.Logger:
    """Return a logger that also writes to the session's debug log file."""
    logger = logging.getLogger(f"eoa.session.{session_id}")
    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(console)

        # File handler (session-specific)
        log_dir = SESSIONS_DIR / session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "processing_debug.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
    return logger
