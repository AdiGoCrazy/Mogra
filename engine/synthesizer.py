import logging
from typing import Any, Optional
import httpx
from config import config
from logger.unified_logger import get_logger, Subsystem

logger = get_logger(Subsystem.ENGINE_SYNTHESIZER)

SYNTHESIS_SYSTEM_PROMPT = """
You are a brilliant local film recommendation expert.
The user provided reference movies they like (seed anchors).
Your task is to recommend ONLY the NEW candidate movies provided in the payload (which do NOT contain the seed movies).

Rules:
1. Match the user's tone.
2. Be direct, punchy, and articulate (max 200 words).
3. Do NOT recommend the seed reference movies back to the user as things to watch. Instead, recommend the NEW candidate movies and explain why they fit the vibe, tone, and genre of the user's favorite reference movies.
"""

class ResponseSynthesizer:
    """Local LLM Synthesizer enforcing active local Ollama execution."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        """Initialize Local LLM client for synthesis.

        Args:
            base_url: Local Ollama base URL.
            model: Local LLM model identifier (e.g. qwen2.5:3b).
        """
        self.base_url = base_url or config.LOCAL_LLM_BASE_URL
        self.model = model or config.LOCAL_LLM_MODEL
        self.api_key = config.LOCAL_LLM_API_KEY

    def _ensure_client(self) -> None:
        """Verify Ollama server connection or raise explicit RuntimeError.

        Raises:
            RuntimeError: If Ollama server is offline or unreachable.
        """
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if resp.status_code != 200:
                raise RuntimeError("Ollama server responded with non-200 status code.")
        except Exception as e:
            raise RuntimeError(
                f"❌ CRITICAL: Local Ollama server is NOT running on port 11434!\n"
                f"Please start Ollama in your terminal using `ollama serve` and make sure `{self.model}` is available.\n"
                f"Error detail: {e}"
            )

    def synthesize_response(
        self,
        user_query: str,
        recommended_movies: list[dict[str, Any]],
        seed_anchors: Optional[list[str]] = None
    ) -> str:
        """Generate response explaining recommendations strictly using Local Ollama with token caps.

        Args:
            user_query: Original user input string.
            recommended_movies: List of recommended candidate movie dictionaries (excluding seed anchors).
            seed_anchors: Optional list of seed reference movie titles.

        Returns:
            Formatted natural language response.

        Raises:
            RuntimeError: If Ollama server is offline or synthesis fails.
        """
        if not recommended_movies:
            return f"No new movie recommendations found matching: '{user_query}'."

        self._ensure_client()

        top_payload = [
            {
                "title": m.get("title"),
                "year": m.get("release_year"),
                "imdb_rating": m.get("imdb_rating"),
                "primary_genre": m.get("primary_genre")
            }
            for m in recommended_movies
        ]

        anchors_str = ", ".join(seed_anchors) if seed_anchors else "None"

        prompt = f"""
        User Query: "{user_query}"
        User Reference Seed Movies (Already Watched): [{anchors_str}]
        New Candidate Recommendations (To Recommend): {top_payload}
        
        Write a concise, engaging response explaining why these NEW candidate movies are great recommendations for someone who likes {anchors_str}.
        """

        try:
            resp = httpx.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"num_predict": 300}
                },
                timeout=30.0
            )
            if resp.status_code == 200:
                res_data = resp.json()
                content = res_data.get("message", {}).get("content", "")
                if content:
                    return content.strip()
            raise RuntimeError(f"Ollama API error (HTTP {resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Local LLM synthesis timeout/error using model '{self.model}': {e}. Returning deterministic candidate summary.")
            titles_str = ", ".join([m.get("title", "") for m in recommended_movies[:3] if m.get("title")])
            return f"Based on your query '{user_query}', here are top recommendations: {titles_str}."

response_synthesizer = ResponseSynthesizer()
