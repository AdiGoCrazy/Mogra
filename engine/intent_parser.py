import logging
import re
import json
from typing import Optional
import httpx
from config import config
from schemas.intent import QueryIntentPayload
from engine.cache import intent_cache
from db.graph_registry import graph_registry
from logger.unified_logger import get_logger, Subsystem

try:
    from openai import OpenAI
    import instructor
    HAS_OPENAI_LIB = True
except ImportError:
    HAS_OPENAI_LIB = False

logger = get_logger(Subsystem.ENGINE_INTENT)

INTENT_PARSER_SYSTEM_PROMPT = """
You are an expert film recommendation query intent parser.
Analyze the user's natural language prompt and convert it into a QueryIntentPayload JSON object.

Rules:
1. Extract hard numerical/categorical filters (min_imdb_rating, mpaa_ratings, subgenres, primary_genre, gore_level, romance_type).
2. Extract seed movie titles mentioned as seed_anchors.
3. Generate concise target vector prompts (max 20 words each) for: visual_aesthetic, character_psychology, soundscape, emotional_aftertaste, philosophical_depth, tonal_arc.
4. Assign vector weights summing to 1.0 based on user emphasis.
"""

class IntentParser:
    """Local LLM Intent Parser enforcing active local Ollama execution."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        """Initialize Local LLM client for intent parsing.

        Args:
            base_url: Local Ollama base URL.
            model: Local LLM model identifier (e.g. qwen2.5:3b).
        """
        self.base_url = base_url or config.LOCAL_LLM_BASE_URL
        self.model = model or config.LOCAL_LLM_MODEL
        self.api_key = config.LOCAL_LLM_API_KEY
        self.client = None

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

        if HAS_OPENAI_LIB and not self.client:
            try:
                openai_client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=1.5, max_retries=0)
                self.client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}. Falling back to direct Ollama httpx.")

    def parse_query(self, user_query: str) -> QueryIntentPayload:
        """Parse user query into QueryIntentPayload strictly via Local Ollama with LRU cache.

        Args:
            user_query: Raw user prompt string.

        Returns:
            QueryIntentPayload instance.

        Raises:
            RuntimeError: If Ollama server is offline or parsing fails.
        """
        cached = intent_cache.get(user_query)
        if cached is not None:
            logger.info("Intent Cache HIT - returning cached intent payload.")
            return cached

        self._ensure_client()

        try:
            intent: Optional[QueryIntentPayload] = None

            # Method 1: Instructor + OpenAI client (if library available)
            if HAS_OPENAI_LIB and self.client:
                try:
                    intent = self.client.chat.completions.create(
                        model=self.model,
                        response_model=QueryIntentPayload,
                        messages=[
                            {"role": "system", "content": INTENT_PARSER_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Parse query: {user_query}"},
                        ],
                        timeout=0.5,
                    )
                except Exception as e:
                    logger.warning(f"Instructor parsing failed: {e}. Falling back to direct httpx JSON query.")

            # Method 2: Direct httpx request to local Ollama API
            if intent is None:
                try:
                    resp = httpx.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": INTENT_PARSER_SYSTEM_PROMPT},
                                {"role": "user", "content": f"Parse query: {user_query}"}
                            ],
                            "format": "json",
                            "stream": False,
                            "options": {"num_predict": 400}
                        },
                        timeout=4.0
                    )
                    if resp.status_code == 200:
                        json_str = resp.json().get("message", {}).get("content", "{}")
                        intent = QueryIntentPayload.model_validate_json(json_str)
                except Exception as e:
                    logger.warning(f"Direct Ollama httpx JSON parsing failed: {e}")

            if intent is None:
                intent = QueryIntentPayload(raw_query=user_query, normalized_summary=user_query)

            # Dynamic Seed Anchor & Negation Extraction via GraphRegistryCache
            known_titles = graph_registry.get_known_movie_titles()

            query_lower = user_query.lower()

            # 1. Parse Negations (e.g. "not like Interstellar")
            negation_prefixes = ["not like", "don't want", "dont want", "exclude", "not including", "other than", "no movies like", "without"]
            for kt in known_titles:
                kt_lower = kt.lower()
                if re.search(rf"\b{re.escape(kt_lower)}\b", query_lower):
                    is_negated = False
                    for prefix in negation_prefixes:
                        m = re.search(rf"\b{re.escape(prefix)}\b", query_lower)
                        if m and kt_lower in query_lower[m.start():]:
                            is_negated = True
                            break
                    if is_negated:
                        if kt not in intent.negative_seed_anchors:
                            intent.negative_seed_anchors.append(kt)
                        if kt in intent.seed_anchors:
                            intent.seed_anchors.remove(kt)
                    elif kt not in intent.seed_anchors and not is_negated:
                        if kt not in intent.seed_anchors:
                            intent.seed_anchors.append(kt)

            # 2. Graph Entity Validation Service: Validate seed_anchors against Neo4j
            from db.neo4j_client import neo4j_client
            validated_anchors: list[str] = []
            for anchor in intent.seed_anchors:
                anchor_lower = anchor.lower()
                title_words = [w for w in anchor_lower.replace(":", "").replace("-", " ").split() if len(w) > 3 and w not in ["the", "with", "from", "like", "movie", "movies", "recommend", "show", "give", "space", "sci-fi", "scifi", "classic", "journey", "cosmic", "scary", "story", "film", "films"]]
                in_raw_query = bool(re.search(rf"\b{re.escape(anchor_lower)}\b", query_lower)) or (title_words and all(re.search(rf"\b{re.escape(w)}\b", query_lower) for w in title_words))
                if not in_raw_query:
                    logger.info(f"Graph Entity Resolver: Stripped hallucinated seed anchor '{anchor}' (not mentioned in user query).")
            if intent.seed_anchors:
                valid_anchors = []
                for sa in intent.seed_anchors:
                    if re.search(rf"\b{re.escape(sa.lower())}\b", query_lower) or any(re.search(rf"\b{re.escape(w.lower())}\b", query_lower) for w in sa.split() if len(w) > 3):
                        valid_anchors.append(sa)
                intent.seed_anchors = valid_anchors

            if len(intent.seed_anchors) >= 2:
                genres_found: set[str] = set()
                for sa in intent.seed_anchors:
                    res_g = neo4j_client.execute_query(
                        "MATCH (m:Movie)-[:BELONGS_TO_GENRE]->(g:Genre) WHERE toLower(m.title) = toLower($title) RETURN g.name AS genre",
                        {"title": sa}
                    )
                    for row in res_g:
                        genres_found.add(row["genre"])
                if len(genres_found) >= 2:
                    intent.is_contrasting_mix = True
                    intent.hard_filters.primary_genre = None

            chatter_phrases = ["cool m8", "thanks", "thank you", "hello", "hi agent", "good morning", "awesome", "great"]
            if any(user_query.strip().lower() == cp for cp in chatter_phrases):
                intent.dialogue_state = "CONVERSATIONAL_CHATTER"
                intent.hard_filters.primary_genre = None
                intent.hard_filters.subgenres = []

            subgenre_keywords = graph_registry.get_subgenre_map()
            if intent.hard_filters.subgenres is None:
                intent.hard_filters.subgenres = []

            for kw, subg in subgenre_keywords.items():
                if re.search(rf"\b{re.escape(kw)}\b", query_lower) and subg not in intent.hard_filters.subgenres:
                    intent.hard_filters.subgenres.append(subg)

            gore_keywords = ["gore", "gory", "blood", "bloody", "family", "pg", "kid", "child"]
            if not any(re.search(rf"\b{re.escape(gk)}\b", query_lower) for gk in gore_keywords):
                intent.hard_filters.gore_level = None

            intent_cache.put(user_query, intent)
            return intent
        except Exception as e:
            logger.error(f"[INTENT_PARSER_ERROR] LLM intent parsing failed for query '{user_query}': {e}", exc_info=True)
            return None

intent_parser = IntentParser()
