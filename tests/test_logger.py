"""Unit tests for Centralized Master Logging System."""

import os
import logging
from logger.unified_logger import configure_logging, get_logger, Subsystem, MASTER_LOG_FILE

def test_unified_logger_initialization() -> None:
    """Verify configure_logging initializes logs/system_master.log file persistence."""
    configure_logging(is_tui=False)
    assert os.path.exists(MASTER_LOG_FILE)

def test_subsystem_logger_formatting() -> None:
    """Verify logger output contains component tag, filename, line number, and thread context."""
    configure_logging(is_tui=False)
    logger = get_logger(Subsystem.ENGINE_INTENT)
    test_msg = "LOG_TEST_UNIQUE_DIAGNOSTIC_MARKER"

    logger.info(test_msg)

    # Read master log file content
    with open(MASTER_LOG_FILE, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert test_msg in log_content
    assert f"[{Subsystem.ENGINE_INTENT}]" in log_content
    assert "[test_logger.py:" in log_content

def test_tui_mode_suppresses_stderr() -> None:
    """Verify is_tui=True suppresses console StreamHandlers from root logger."""
    configure_logging(is_tui=True)
    root_logger = logging.getLogger()

    stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    assert len(stream_handlers) == 0
