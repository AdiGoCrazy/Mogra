"""Centralized Master Logging System for Mogra Movie Recommender Agent.

Provides context-tagged subsystem logging, source file/line identification, thread tracking,
and unified persistence to logs/system_master.log for automated deep diagnostics.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

LOG_DIR = os.path.abspath("logs")
MASTER_LOG_FILE = os.path.join(LOG_DIR, "system_master.log")

# Subsystem Context Namespaces
class Subsystem:
    ENGINE_INTENT = "ENGINE.INTENT_PARSER"
    ENGINE_RETRIEVAL = "ENGINE.HYBRID_RETRIEVER"
    ENGINE_SYNTHESIZER = "ENGINE.SYNTHESIZER"
    ENGINE_EMBEDDINGS = "ENGINE.LOCAL_EMBEDDINGS"
    DB_NEO4J = "DATABASE.NEO4J"
    DB_QDRANT = "DATABASE.QDRANT"
    DB_REGISTRY = "DATABASE.GRAPH_REGISTRY"
    DB_FEEDBACK = "DATABASE.FEEDBACK_STORE"
    API_MAIN = "API.MAIN"
    API_ROUTER = "API.ROUTER"
    TUI_MAIN = "TUI.MAIN"
    TUI_FEEDBACK = "TUI.FEEDBACK_HARNESS"

_logging_configured = False

def configure_logging(is_tui: bool = False, level: int = logging.INFO) -> None:
    """Configure unified logging infrastructure across all modules and third-party libraries.

    Args:
        is_tui: If True, suppresses console StreamHandlers to protect Textual ANSI drawing buffers.
        level: Base logging level (default: logging.INFO).
    """
    global _logging_configured
    os.makedirs(LOG_DIR, exist_ok=True)

    # Standardized deep-diagnostics formatter
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] [Thread-%(thread)d] %(message)s"
    formatter = logging.Formatter(log_format)

    # Master Rotating File Handler (10MB per log file, keeping 5 backups)
    file_handler = RotatingFileHandler(
        MASTER_LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)

    if not is_tui:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root_logger.addHandler(stream_handler)

    # Suppress stderr leaks from noisy third-party libraries
    monitored_third_party = [
        "httpx", "httpcore", "instructor", "qdrant_client", "neo4j", "urllib3", "uvicorn"
    ]
    for lib_name in monitored_third_party:
        lg = logging.getLogger(lib_name)
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        lg.addHandler(file_handler)
        lg.propagate = False

    _logging_configured = True
    root_logger.info(f"[SYSTEM_LOGGING_INIT] Centralized logging initialized -> Master Log: {MASTER_LOG_FILE}")

def get_logger(subsystem_name: str) -> logging.Logger:
    """Factory function returning a context-tagged logger.

    Args:
        subsystem_name: Subsystem tag string (e.g. Subsystem.ENGINE_INTENT).

    Returns:
        logging.Logger instance.
    """
    if not _logging_configured:
        configure_logging()
    return logging.getLogger(subsystem_name)
