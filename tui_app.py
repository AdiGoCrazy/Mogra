"""Textual Terminal User Interface (TUI) Application for Movie Recommendation Agent with Execution Logging."""

import os
import time
import logging
from typing import Any
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.widgets import Button, Footer, Header, Input
from engine.intent_parser import intent_parser
from engine.retrieval import hybrid_retriever
from engine.synthesizer import response_synthesizer
from ui.widgets import AgentMessageWidget, LoadingBannerWidget, MovieCardWidget, TelemetryChartWidget, UserMessageWidget

from logger.unified_logger import configure_logging, get_logger, Subsystem, MASTER_LOG_FILE

configure_logging(is_tui=True)
logger = get_logger(Subsystem.TUI_MAIN)

class MovieAgentApp(App[None]):
    """Textual App providing a minimalist single-pane chat interface for movie recommendations."""

    TITLE = "Movie Recommendation Agent"
    SUB_TITLE = "GraphRAG + Multi-Vector AI Agent"
    CSS_PATH = "tui_styles.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear Chat")
    ]

    def compose(self) -> ComposeResult:
        """Compose child widgets."""
        yield Header(show_clock=True)
        yield ScrollableContainer(id="chat-container")
        with Horizontal(id="input-container"):
            yield Input(
                placeholder="Ask for recommendations (e.g. 'movies like Blade Runner or Alien')...",
                id="user-input"
            )
            yield Button("Send", id="send-button", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """App launch event handler."""
        logger.info(f"[SYSTEM_STARTUP] TUI MovieAgentApp started. Logging to master log: {MASTER_LOG_FILE}")
        chat_container = self.query_one("#chat-container", ScrollableContainer)
        welcome_text = (
            "Welcome to the **Movie Recommendation Agent**!\n\n"
            "Ask for movie recommendations by genre, mood, or seed references.\n"
            "Example: *'movies like Blade Runner'*, *'scary creature horror in space'*"
        )
        chat_container.mount(AgentMessageWidget(welcome_text))

    def on_key(self, event: Any) -> None:
        """Log key press events cleanly without flooding log files."""
        key_name = str(getattr(event, 'key', '')).lower()
        if key_name in ("ctrl+l", "q", "enter", "escape", "tab"):
            focused_widget = self.focused
            focused_id = focused_widget.id if (focused_widget and hasattr(focused_widget, 'id') and focused_widget.id) else "None"
            logger.info(f"[KEY_PRESS] Shortcut/Nav Key: '{key_name}' | Focused Widget: '{focused_id}'")

    def action_clear_chat(self) -> None:
        """Clear all chat history messages."""
        logger.info("[USER_ACTION] User cleared chat history via Ctrl+L shortcut.")
        chat_container = self.query_one("#chat-container", ScrollableContainer)
        chat_container.remove_children()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Send button press."""
        btn_id = event.button.id or "Unknown"
        btn_label = str(event.button.label)
        logger.info(f"[BUTTON_PRESS] On-screen button pressed: '{btn_label}' (ID: '{btn_id}')")
        if event.button.id == "send-button":
            self._handle_user_submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key press in input field."""
        input_id = event.input.id or "Unknown"
        logger.info(f"[INPUT_SUBMITTED] Input field '{input_id}' submitted via Enter key: '{event.value}'")
        if event.input.id == "user-input":
            self._handle_user_submit()

    def _handle_user_submit(self) -> None:
        """Process user input submission."""
        user_input = self.query_one("#user-input", Input)
        query_text = user_input.value.strip()
        if not query_text:
            return

        logger.info(f"[USER_INPUT] User prompt submitted: '{query_text}'")
        user_input.value = ""
        chat_container = self.query_one("#chat-container", ScrollableContainer)

        # 1. Append User Message Bubble
        chat_container.mount(UserMessageWidget(query_text))

        # 2. Append Loading Banner
        loading_widget = LoadingBannerWidget()
        chat_container.mount(loading_widget)
        chat_container.scroll_end(animate=False)

        # 3. Launch background worker in a separate OS thread via lambda
        self.run_worker(
            lambda: self._query_worker(query_text, loading_widget),
            thread=True,
            exclusive=False
        )

    def _query_worker(self, query_text: str, loading_widget: LoadingBannerWidget) -> None:
        """Background worker thread processing intent parsing, retrieval, and synthesis."""
        start_time = time.time()
        logger.info(f"[WORKER_START] Query processing initiated in thread for: '{query_text}'")

        try:
            # Step A: Local LLM Intent Parsing
            t0 = time.time()
            intent = intent_parser.parse_query(query_text)
            dt_intent = time.time() - t0

            if not intent:
                logger.error(f"[QUERY_ERROR] Intent parser returned None for query: '{query_text}'")
                explanation = "Sorry, I could not find the movie requested."
                metrics = {"parse_ms": dt_intent * 1000.0, "retrieval_ms": 0.0, "total_ms": (time.time() - start_time) * 1000.0, "candidate_count": 0}
                self.call_from_thread(self._render_agent_response, explanation, [], metrics, loading_widget)
                return

            logger.info(
                f"[INTENT_PARSED] ({dt_intent:.2f}s) Seed Anchors: {intent.seed_anchors} | "
                f"Primary Genre: {intent.hard_filters.primary_genre} | "
                f"Subgenres: {intent.hard_filters.subgenres}"
            )

            # Step B: Retrieval & Candidate Graph Encodings with Vector Similarity Thresholding
            t0 = time.time()
            raw_candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=1, min_similarity_threshold=0.30)

            # Enrich candidates with synopsis & subgenres from Neo4j
            enriched_candidates: list[dict[str, Any]] = []
            for c in raw_candidates:
                tmdb_id = c["tmdb_id"]
                cypher = """
                MATCH (m:Movie {tmdb_id: $tmdb_id})
                OPTIONAL MATCH (m)-[:HAS_SUBGENRE]->(s:Subgenre)
                OPTIONAL MATCH (m)-[:BELONGS_TO_GENRE]->(g:Genre)
                RETURN m.title AS title, m.release_year AS release_year, m.imdb_rating AS imdb_rating,
                       m.mpaa_rating AS mpaa_rating, m.narrative_synopsis AS synopsis,
                       g.name AS primary_genre, collect(s.name) AS subgenres
                """
                res = hybrid_retriever.neo4j.execute_query(cypher, {"tmdb_id": tmdb_id})
                if res:
                    enriched_candidates.append(res[0])
                else:
                    enriched_candidates.append(c)

            dt_retrieval = time.time() - t0
            candidate_titles = [c.get("title", "Unknown") for c in enriched_candidates]
            logger.info(f"[GRAPH_RETRIEVAL] ({dt_retrieval:.2f}s) Top Thresholded Candidates: {candidate_titles}")

            if not enriched_candidates:
                logger.error(f"[QUERY_ERROR] Candidate retrieval found 0 candidates for query: '{query_text}'")
                explanation = "Sorry, I could not find the movie requested."
                metrics = {"parse_ms": dt_intent * 1000.0, "retrieval_ms": dt_retrieval * 1000.0, "total_ms": (time.time() - start_time) * 1000.0, "candidate_count": 0}
                self.call_from_thread(self._render_agent_response, explanation, [], metrics, loading_widget)
                return

            # Step C: Local LLM Response Synthesis
            t0 = time.time()
            explanation = response_synthesizer.synthesize_response(
                query_text,
                enriched_candidates,
                seed_anchors=intent.seed_anchors
            )
            dt_synthesis = time.time() - t0
            total_dt = time.time() - start_time

            logger.info(
                f"[LLM_SYNTHESIS] ({dt_synthesis:.2f}s) Response Generated (Total Latency: {total_dt:.2f}s):\n"
                f"{explanation[:200]}..."
            )

            # Thread-safe UI update back to main thread
            metrics = {
                "parse_ms": dt_intent * 1000.0,
                "retrieval_ms": dt_retrieval * 1000.0,
                "total_ms": total_dt * 1000.0,
                "candidate_count": len(enriched_candidates)
            }
            self.call_from_thread(self._render_agent_response, explanation, enriched_candidates, metrics, loading_widget)

        except Exception as e:
            logger.error(f"[WORKER_ERROR] Failed query processing in thread: {e}", exc_info=True)
            self.call_from_thread(self._handle_worker_error, str(e), loading_widget)

    def _render_agent_response(
        self,
        explanation: str,
        candidates: list[dict[str, Any]],
        metrics: dict[str, float],
        loading_widget: LoadingBannerWidget
    ) -> None:
        """UI update function called on main thread to render agent output, cards, and telemetry charts."""
        chat_container = self.query_one("#chat-container", ScrollableContainer)

        # Remove loading banner
        loading_widget.remove()

        # Mount Agent Prose Response
        chat_container.mount(AgentMessageWidget(explanation))

        # Mount Movie Recommendation Cards
        for rank, movie in enumerate(candidates, start=1):
            chat_container.mount(MovieCardWidget(movie, rank=rank))

        # Mount Telemetry Chart Widget
        chat_container.mount(
            TelemetryChartWidget(
                parse_ms=metrics.get("parse_ms", 0.0),
                retrieval_ms=metrics.get("retrieval_ms", 0.0),
                total_ms=metrics.get("total_ms", 0.0),
                candidate_count=int(metrics.get("candidate_count", 0))
            )
        )

        chat_container.scroll_end(animate=True)
        user_input = self.query_one("#user-input", Input)
        user_input.focus()
        logger.info("[UI_RENDERED] Rendered Agent response and recommendation cards on chat UI.")

    def _handle_worker_error(self, error_msg: str, loading_widget: LoadingBannerWidget) -> None:
        """UI update function handling worker exceptions."""
        chat_container = self.query_one("#chat-container", ScrollableContainer)
        loading_widget.remove()
        error_text = f"❌ Error processing query: {error_msg}"
        chat_container.mount(AgentMessageWidget(error_text))
        chat_container.scroll_end(animate=True)
        user_input = self.query_one("#user-input", Input)
        user_input.focus()

if __name__ == "__main__":
    app = MovieAgentApp()
    app.run()
