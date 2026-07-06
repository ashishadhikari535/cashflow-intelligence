"""
utils/logger.py
Structured logging for the Cash Flow Intelligence System.
"""

import sys
from pathlib import Path

from loguru import logger

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_LABEL = ROOT_DIR.name

# Avoid UnicodeEncodeError on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    level="INFO",
    colorize=True,
)

logger.add(
    LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)


def get_logger(name: str):
    """Return a module-bound logger."""
    return logger.bind(name=name)


def format_path(path: str | Path) -> str:
    """
    Render paths relative to project root for portable logs.
    Example: cashflow-intelligence/data/exports/file.png
    """
    p = Path(path).resolve()
    try:
        rel = p.relative_to(ROOT_DIR)
        return f"{PROJECT_LABEL}/{rel.as_posix()}"
    except ValueError:
        return p.as_posix()
