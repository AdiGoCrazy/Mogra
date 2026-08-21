"""Custom Textual widgets for the Movie Recommendation Agent TUI."""

from typing import Any, Optional
from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Markdown

class UserMessageWidget(Widget):
    """Widget displaying a user query message."""
    can_focus = False

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        content = Text.assemble(
            ("👤 User: ", "bold cyan"),
            (self.message, "white")
        )
        s = Static(Panel(content, border_style="cyan", title="Prompt", expand=False))
        s.can_focus = False
        yield s

class LoadingBannerWidget(Widget):
    """Widget displaying an active background worker loading indicator."""
    can_focus = False

    def compose(self) -> ComposeResult:
        content = Text("🧠 Searching GraphRAG & Qdrant vectors... Please wait...", style="bold yellow")
        s = Static(Panel(content, border_style="yellow", expand=False))
        s.can_focus = False
        yield s

class AgentMessageWidget(Widget):
    """Widget displaying agent synthesis response formatted as clean Markdown."""
    can_focus = False

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        m = Markdown(self.message, classes="agent-prose")
        m.can_focus = False
        yield m

class MovieCardWidget(Widget):
    """Widget displaying a clean formatted Rich movie recommendation card."""
    can_focus = False

    def __init__(self, movie_data: dict[str, Any], rank: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.movie_data = movie_data
        self.rank = rank

    def compose(self) -> ComposeResult:
        title = self.movie_data.get("title", "Unknown Title")
        year = self.movie_data.get("release_year", "N/A")
        imdb = self.movie_data.get("imdb_rating", "N/A")
        mpaa = self.movie_data.get("mpaa_rating", "NR")
        primary_genre = self.movie_data.get("primary_genre", "N/A")
        subgenres = self.movie_data.get("subgenres", [])
        sub_str = ", ".join(subgenres) if subgenres else "N/A"
        synopsis = self.movie_data.get("synopsis", "")

        card_text = Text()
        card_text.append(f"🎬 Rank #{self.rank}: ", style="bold gold1")
        card_text.append(f"{title} ", style="bold white")
        card_text.append(f"({year})\n", style="bright_white")

        card_text.append("⭐ IMDb: ", style="bold yellow")
        card_text.append(f"{imdb}/10  ", style="bold white")
        card_text.append("🔒 Rated: ", style="bold magenta")
        card_text.append(f"{mpaa}\n", style="white")

        card_text.append("🏷️ Genre: ", style="bold cyan")
        card_text.append(f"{primary_genre} ", style="bold white")
        card_text.append(f"[{sub_str}]\n", style="italic cyan")

        if synopsis:
            card_text.append("\n📖 Synopsis:\n", style="bold green")
            card_text.append(synopsis, style="white")

        panel = Panel(
            card_text,
            border_style="gold1",
            title=f"[bold gold1]Movie Recommendation #{self.rank}[/bold gold1]",
            subtitle=f"[bold white]{title}[/bold white]",
            expand=False
        )
        s = Static(panel)
        s.can_focus = False
        yield s

class TelemetryChartWidget(Widget):
    """Widget rendering bulletproof visual ASCII telemetry bar charts for query latency and performance."""
    can_focus = False

    def __init__(
        self,
        parse_ms: float = 0.0,
        retrieval_ms: float = 0.0,
        total_ms: float = 0.0,
        candidate_count: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.parse_ms = parse_ms
        self.retrieval_ms = retrieval_ms
        self.total_ms = total_ms
        self.candidate_count = candidate_count

    def compose(self) -> ComposeResult:
        def build_bar(val_ms: float, max_ms: float = 1000.0, width: int = 16) -> str:
            if max_ms <= 0:
                ratio = 0.0
            else:
                ratio = min(1.0, max(0.0, val_ms / max_ms))
            filled = int(round(ratio * width))
            empty = width - filled
            return f"[{'█' * filled}{'░' * empty}] {val_ms:5.1f} ms"

        max_scale = max(self.total_ms, 100.0)
        chart_text = Text()
        chart_text.append("📊 QUERY TELEMETRY & LATENCY BREAKDOWN\n", style="bold cyan")
        chart_text.append("────────────────────────────────────────────\n", style="bright_black")

        chart_text.append("⚡ Intent Parsing   : ", style="bold yellow")
        chart_text.append(f"{build_bar(self.parse_ms, max_scale)}\n", style="bold green")

        chart_text.append("🔍 Vector Retrieval : ", style="bold yellow")
        chart_text.append(f"{build_bar(self.retrieval_ms, max_scale)}\n", style="bold green")

        chart_text.append("⏱️ Total Latency    : ", style="bold yellow")
        chart_text.append(f"{build_bar(self.total_ms, max_scale)}\n", style="bold green")

        chart_text.append("────────────────────────────────────────────\n", style="bright_black")
        chart_text.append(f"🎯 Candidates Retrieved: ", style="bold white")
        chart_text.append(f"{self.candidate_count} movies\n", style="bold magenta")

        panel = Panel(
            chart_text,
            border_style="cyan",
            title="[bold cyan]Telemetry Metrics[/bold cyan]",
            expand=False
        )
        s = Static(panel)
        s.can_focus = False
        yield s
