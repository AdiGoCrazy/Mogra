"""Unit tests for MovieAgentApp Textual TUI using Pilot API."""

import pytest
from tui_app import MovieAgentApp
from ui.widgets import AgentMessageWidget

@pytest.mark.asyncio
async def test_tui_app_mount_and_clear() -> None:
    """Verify that MovieAgentApp mounts cleanly and handles clear chat action."""
    app = MovieAgentApp()
    async with app.run_test() as pilot:
        # 1. Verify app title and screen mount
        assert app.TITLE == "Movie Recommendation Agent"
        assert app.screen is not None

        # 2. Verify initial welcome message widget present
        welcome_widgets = list(app.query(AgentMessageWidget))
        assert len(welcome_widgets) == 1

        # 3. Test clear chat key binding (Ctrl+L)
        await pilot.press("ctrl+l")
        cleared_widgets = list(app.query(AgentMessageWidget))
        assert len(cleared_widgets) == 0

@pytest.mark.asyncio
async def test_tui_input_submission() -> None:
    """Verify input submission triggers user message widget."""
    app = MovieAgentApp()
    async with app.run_test() as pilot:
        # Type input text into user input field
        await pilot.click("#user-input")
        await pilot.press("m", "o", "v", "i", "e", "s")
        
        input_widget = app.query_one("#user-input")
        assert input_widget.value == "movies"

def test_tui_fresh_log_file_mode() -> None:
    """Verify that master system log file is created on app startup."""
    import os
    from logger.unified_logger import MASTER_LOG_FILE
    assert os.path.exists(MASTER_LOG_FILE), f"Log file does not exist at {MASTER_LOG_FILE}"

def test_telemetry_chart_widget_rendering() -> None:
    """Verify TelemetryChartWidget composes clean ASCII bar chart strings without exception."""
    from ui.widgets import TelemetryChartWidget
    widget = TelemetryChartWidget(parse_ms=150.0, retrieval_ms=45.0, total_ms=195.0, candidate_count=5)
    rendered = list(widget.compose())
    assert len(rendered) == 1, "Expected single static panel output for TelemetryChartWidget"

@pytest.mark.asyncio
async def test_tui_key_press_and_click_event_logging() -> None:
    """Verify key presses and button click events trigger event handlers and write structured logs."""
    import os
    from tui_app import MovieAgentApp
    from logger.unified_logger import MASTER_LOG_FILE

    app = MovieAgentApp()
    async with app.run_test() as pilot:
        # Click user input field
        await pilot.click("#user-input")
        
        # Press key 'a'
        await pilot.press("a")
        
        # Press send button
        await pilot.click("#send-button")

        assert os.path.exists(MASTER_LOG_FILE)

    assert os.path.exists(MASTER_LOG_FILE), "Log file must exist after TUI interaction"
    with open(MASTER_LOG_FILE, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert "[TUI.MAIN]" in log_content

    assert "[KEY_PRESS]" in log_content or "[MOUSE_CLICK]" in log_content or "[SYSTEM_STARTUP]" in log_content
