"""Hybrid retrieval engine combining Neo4j Cypher pre-filtering with Qdrant Multi-Vector search and RRF thresholding."""

import logging
import re
from typing import Any, Optional
from db.neo4j_client import Neo4jClient
from db.qdrant_client import QdrantClientWrapper
from schemas.intent import QueryIntentPayload
from engine.local_embeddings import local_embedder
from config import config
from db.graph_registry import graph_registry
from logger.unified_logger import get_logger, Subsystem

logger = get_logger(Subsystem.ENGINE_RETRIEVAL)

def normalize_genre_names(raw_genre: str) -> list[str]:
    """Expand raw genre string into list of canonical database names and aliases.

    Args:
        raw_genre: Raw genre string (e.g. 'Sci-Fi').

    Returns:
        List of matching lower-case string names.
    """
    genre_alias_map = graph_registry.get_genre_alias_map()
    key = (raw_genre or "").strip().lower().replace("_", " ")
    raw_key = (raw_genre or "").strip().lower()
    aliases = genre_alias_map.get(raw_key, genre_alias_map.get(key, [raw_genre]))
    return [a.lower() for a in aliases]

def extract_setting_keywords(user_query: str) -> list[str]:
    """Extract spatial setting keywords from raw user prompt string.

    Args:
        user_query: Raw user prompt string.

    Returns:
        List of matching setting tag strings.
    """
    setting_keyword_map = graph_registry.get_setting_keyword_map()
    query_lower = (user_query or "").lower()
    matched_settings: list[str] = []
    for kw, tags in setting_keyword_map.items():
        if re.search(rf"\b{re.escape(kw)}\b", query_lower):
            matched_settings.extend(tags)
    return list(set(matched_settings))

class HybridRetrievalEngine:
    """Orchestrator for GraphRAG pre-filtering, multi-vector retrieval, and RRF score fusion."""

    def __init__(
        self,
        neo4j: Optional[Neo4jClient] = None,
        qdrant: Optional[QdrantClientWrapper] = None
    ) -> None:
        """Initialize retrieval engine database clients.

        Args:
            neo4j: Neo4j database client wrapper.
            qdrant: Qdrant vector database wrapper.
        """
        from db.neo4j_client import neo4j_client
        from db.qdrant_client import qdrant_wrapper
        self.neo4j = neo4j or neo4j_client
        self.qdrant = qdrant or qdrant_wrapper

    def build_cypher_query(self, intent: QueryIntentPayload, tier: int = 1) -> tuple[str, dict[str, Any]]:
        """Construct Cypher query based on intent hard filters, settings, and relaxation tier.

        Args:
            intent: Parsed query intent payload.
            tier: Progressive Constraint Relaxation tier (1=strict, 2=relaxed, 3=broad).

        Returns:
            Tuple of (Cypher query string, parameters dictionary).
        """
        filters = intent.hard_filters
        where_clauses: list[str] = []
        params: dict[str, Any] = {}

        # 1. Primary Genre & Subgenre alias handling
        genre_list: list[str] = []
        if filters.primary_genre:
            genre_list.extend(normalize_genre_names(filters.primary_genre))
        if filters.subgenres:
            for sg in filters.subgenres:
                genre_list.extend(normalize_genre_names(sg))

        if genre_list:
            where_clauses.append("toLower(g.name) IN $genre_list")
            params["genre_list"] = list(set(genre_list))

        # 2. Spatial Setting Keywords Handling
        setting_tags = extract_setting_keywords(intent.raw_query)
        if setting_tags and tier in (1, 2):
            where_clauses.append("toLower(sett.name) IN $setting_tags")
            params["setting_tags"] = setting_tags

        # 3. Rating filters
        if filters.min_imdb_rating and tier == 1:
            where_clauses.append("m.imdb_rating >= $min_imdb")
            params["min_imdb"] = filters.min_imdb_rating
        elif filters.min_imdb_rating and tier == 2:
            where_clauses.append("m.imdb_rating >= $min_imdb")
            params["min_imdb"] = max(0.0, filters.min_imdb_rating - 1.0)

        if filters.mpaa_ratings and tier in (1, 2):
            where_clauses.append("m.mpaa_rating IN $mpaa_ratings")
            params["mpaa_ratings"] = filters.mpaa_ratings

        if filters.gore_level and tier == 1:
            where_clauses.append("m.gore_level = $gore_level")
            params["gore_level"] = filters.gore_level

        # 4. Negative Exclusions Handling
        negated_titles = [t.strip().lower() for t in (intent.negative_seed_anchors + intent.negative_exclusions) if t]
        if negated_titles:
            where_clauses.append("NOT toLower(m.title) IN $negated_titles")
            params["negated_titles"] = negated_titles

        if filters.has_jump_scares is False:
            where_clauses.append("m.has_jump_scares = false")

        if filters.has_romance is False:
            where_clauses.append("m.has_romance = false")

        where_str = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Use mandatory MATCH for SET_IN when spatial setting tags are present
        if setting_tags and tier in (1, 2):
            cypher = f"""
            MATCH (m:Movie)
            MATCH (m)-[:BELONGS_TO_GENRE|HAS_SUBGENRE]->(g)
            MATCH (m)-[:SET_IN]->(sett:Setting)
            WHERE {where_str}
            RETURN DISTINCT m.tmdb_id AS tmdb_id, m.title AS title
            LIMIT {config.CYPHER_MAX_LIMIT}
            """
        else:
            cypher = f"""
            MATCH (m:Movie)
            MATCH (m)-[:BELONGS_TO_GENRE|HAS_SUBGENRE]->(g)
            WHERE {where_str}
            RETURN DISTINCT m.tmdb_id AS tmdb_id, m.title AS title
            LIMIT {config.CYPHER_MAX_LIMIT}
            """
        return cypher, params

    def execute_progressive_cypher(self, intent: QueryIntentPayload, min_candidates: int = 15) -> list[int]:
        """Execute Cypher query with Progressive Constraint Relaxation if candidate pool < min_candidates.

        Args:
            intent: Parsed query intent.
            min_candidates: Minimum acceptable candidate pool size.

        Returns:
            List of candidate TMDB ID integers.
        """
        for tier in (1, 2, 3):
            cypher, params = self.build_cypher_query(intent, tier=tier)
            results = self.neo4j.execute_query(cypher, params)
            candidate_ids = [r["tmdb_id"] for r in results if r.get("tmdb_id") is not None]

            logger.info(f"Cypher PCR Tier {tier} returned {len(candidate_ids)} candidate IDs.")
            if len(candidate_ids) >= min_candidates:
                return candidate_ids

        # Fail-Safe Recovery: If 0 records returned, fallback to empty list so cypher_allowed_ids = None (permitting Vector search salvage)
        logger.warning(f"Cypher Hard Pre-Filter returned 0 candidate IDs for genre '{intent.hard_filters.primary_genre}'. Falling back to broad Qdrant Vector search to prevent candidate lockout.")
        return []

    def retrieve_recommendations(
        self,
        intent: QueryIntentPayload,
        top_k: int = 1,
        min_similarity_threshold: float = 0.35
    ) -> list[dict[str, Any]]:
        """Retrieve candidate recommendations enforcing Qdrant multi-vector distance scoring and RRF thresholding.

        Args:
            intent: Parsed query intent payload.
            top_k: Number of recommendations to return.
            min_similarity_threshold: Minimum acceptable cosine similarity score.

        Returns:
            List of candidate movie dictionaries meeting threshold.
        """
        seed_anchors_lower = [s.strip().lower() for s in intent.seed_anchors if s]
        sequel_candidates: list[dict[str, Any]] = []

        # 0. Execute Cypher hard pre-filtering if primary genre, subgenres, or setting tags specified
        cypher_allowed_ids: Optional[set[int]] = None
        setting_tags = extract_setting_keywords(intent.raw_query)

        if intent.is_contrasting_mix and len(seed_anchors_lower) >= 2:
            union_cypher = """
            MATCH (anchor:Movie)
            WHERE toLower(anchor.title) IN $anchors
            MATCH (anchor)-[:BELONGS_TO_GENRE|HAS_SUBGENRE]->(g)
            MATCH (m:Movie)-[:BELONGS_TO_GENRE|HAS_SUBGENRE]->(g)
            RETURN DISTINCT m.tmdb_id AS tmdb_id
            """
            union_res = self.neo4j.execute_query(union_cypher, {"anchors": seed_anchors_lower})
            if union_res:
                cypher_allowed_ids = {r["tmdb_id"] for r in union_res}
                logger.info(f"Multi-Anchor Hybrid Fusion: Graph Union across anchors {seed_anchors_lower} -> {len(cypher_allowed_ids)} allowed IDs.")
        elif intent.hard_filters.primary_genre or intent.hard_filters.subgenres or setting_tags:
            allowed_ids_list = self.execute_progressive_cypher(intent, min_candidates=1)
            if allowed_ids_list:
                cypher_allowed_ids = set(allowed_ids_list)
                logger.info(f"Hard Filter Enforced ({intent.hard_filters.primary_genre}): {len(cypher_allowed_ids)} allowed IDs -> {allowed_ids_list}")
            else:
                logger.error("[RETRIEVAL_ERROR] Hard filter constraints resulted in 0 candidate movie IDs. Returning empty candidate set.")
                return []

        # 1. Query Graph for Direct Sequels of Seed Anchors
        if seed_anchors_lower:
            sequel_cypher = """
            MATCH (m:Movie)-[:IS_SEQUEL_TO]->(parent:Movie)
            OPTIONAL MATCH (m)-[:BELONGS_TO_GENRE]->(g:Genre)
            RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.imdb_rating AS imdb_rating, m.release_year AS release_year, g.name AS primary_genre, parent.title AS parent_title
            """
            sequel_results = self.neo4j.execute_query(sequel_cypher)
            for seq in sequel_results:
                parent_title = (seq.get("parent_title") or "").lower()
                if any(anchor in parent_title or parent_title in anchor for anchor in seed_anchors_lower):
                    logger.info(f"Graph MATCH: Found direct sequel '{seq['title']}' for seed anchor '{seq['parent_title']}'!")
                    sequel_candidates.append(seq)

        # 2. Vector Search & RRF Scoring across Qdrant Named Vectors using search(query_vector=(name, vec))
        candidate_ranks: dict[int, dict[str, int]] = {}
        max_scores: dict[int, float] = {}

        vec_mapping = {
            "raw_query": intent.raw_query,
            "visual_aesthetic": intent.vector_prompts.visual_aesthetic_prompt,
            "character_psychology": intent.vector_prompts.character_psychology_prompt,
            "emotional_aftertaste": intent.vector_prompts.emotional_aftertaste_prompt,
            "soundscape": intent.vector_prompts.soundscape_prompt,
            "philosophical_depth": intent.vector_prompts.philosophical_depth_prompt,
            "tonal_arc": intent.vector_prompts.tonal_arc_prompt,
            "dialogue_and_wit": getattr(intent.vector_prompts, "dialogue_and_wit_prompt", ""),
            "pacing_and_kinetic_rhythm": getattr(intent.vector_prompts, "pacing_and_kinetic_rhythm_prompt", ""),
            "spatial_atmosphere": getattr(intent.vector_prompts, "spatial_atmosphere_prompt", ""),
            "cultural_historical_texture": getattr(intent.vector_prompts, "cultural_historical_texture_prompt", ""),
            "climactic_catharsis": getattr(intent.vector_prompts, "climactic_catharsis_prompt", ""),
            "antagonist_threat_dynamics": getattr(intent.vector_prompts, "antagonist_threat_dynamics_prompt", ""),
            "thematic_subtext_allegory": getattr(intent.vector_prompts, "thematic_subtext_allegory_prompt", ""),
            "humor_and_irony_tone": getattr(intent.vector_prompts, "humor_and_irony_tone_prompt", ""),
            "intimacy_and_chemistry": getattr(intent.vector_prompts, "intimacy_and_chemistry_prompt", ""),
            "dread_suspense_escalation": getattr(intent.vector_prompts, "dread_suspense_escalation_prompt", ""),
        }

        try:
            qdrant_client = self.qdrant.get_client()
            vector_weights = intent.vector_weights.model_dump()
            vector_weights["raw_query"] = 0.25  # Give strong weight to raw query embedding

            raw_emb = local_embedder.embed_text(intent.raw_query)

            for vec_name, prompt_text in vec_mapping.items():
                if vec_name == "raw_query":
                    emb = raw_emb
                    using_vec = "visual_aesthetic"
                elif prompt_text:
                    if prompt_text == intent.raw_query:
                        emb = raw_emb
                    else:
                        emb = local_embedder.embed_text(prompt_text)
                    using_vec = vec_name
                else:
                    continue

                # Use search API compatible with Qdrant server 1.8.3
                search_res = qdrant_client.search(
                    collection_name=config.QDRANT_COLLECTION,
                    query_vector=(using_vec, emb),
                    limit=config.VECTOR_SEARCH_LIMIT
                )

                for rank_idx, hit in enumerate(search_res):
                    point_id = hit.id
                    score = hit.score

                    if point_id not in candidate_ranks:
                        candidate_ranks[point_id] = {}
                        max_scores[point_id] = 0.0

                    candidate_ranks[point_id][vec_name] = rank_idx
                    if score > max_scores[point_id]:
                        max_scores[point_id] = score

            rrf_sorted = self.compute_rrf_scores(candidate_ranks, vector_weights)

        except Exception as e:
            logger.warning(f"Qdrant vector search failed, falling back to Cypher candidates: {e}", exc_info=True)
            rrf_sorted = []

        # 3. Assemble Candidates (Sequels first, then vector-ranked candidates meeting threshold)
        final_candidates: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        for seq in sequel_candidates:
            tmdb_id = seq["tmdb_id"]
            if tmdb_id not in seen_ids:
                final_candidates.append(seq)
                seen_ids.add(tmdb_id)

        # Process vector-ranked candidates
        for tmdb_id, rrf_score in rrf_sorted:
            if len(final_candidates) >= top_k:
                break

            if tmdb_id in seen_ids:
                continue

            # Check Cypher hard filter constraint
            if cypher_allowed_ids is not None and tmdb_id not in cypher_allowed_ids:
                logger.info(f"Hard Filter Filtered Out: ID {tmdb_id} not in allowed genre/setting list")
                continue

            max_sim = max_scores.get(tmdb_id, 0.0)

            # Query movie metadata from Neo4j
            res = self.neo4j.execute_query(
                "MATCH (m:Movie {tmdb_id: $id}) OPTIONAL MATCH (m)-[:BELONGS_TO_GENRE]->(g:Genre) RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.imdb_rating AS imdb_rating, m.release_year AS release_year, g.name AS primary_genre",
                {"id": tmdb_id}
            )
            if not res:
                continue

            movie_info = res[0]
            title = movie_info.get("title", "")
            title_lower = title.strip().lower()

            # Exclude exact seed anchors
            if any(anchor == title_lower or anchor == title_lower.replace(":", "") for anchor in seed_anchors_lower):
                logger.info(f"Filtering out seed reference anchor movie: '{title}'")
                continue

            # Exclude negated seed anchors & negative exclusions
            negated_titles_lower = [t.strip().lower() for t in (intent.negative_seed_anchors + intent.negative_exclusions) if t]
            if any(neg in title_lower or title_lower in neg for neg in negated_titles_lower):
                logger.info(f"Filtering out negative reference anchor movie: '{title}'")
                continue

            # Enforce similarity threshold filtering
            if max_sim < min_similarity_threshold:
                logger.info(f"Threshold Filter: Excluding '{title}' (Cosine Sim: {max_sim:.3f} < {min_similarity_threshold:.2f})")
                continue

            logger.info(f"Threshold Passed: Retaining '{title}' (Cosine Sim: {max_sim:.3f}, RRF: {rrf_score:.4f})")
            final_candidates.append(movie_info)
            seen_ids.add(tmdb_id)

        return final_candidates[:top_k]

    def compute_rrf_scores(
        self,
        candidate_ranks: dict[int, dict[str, int]],
        weights: dict[str, float],
        k: Optional[int] = None
    ) -> list[tuple[int, float]]:
        """Calculate Reciprocal Rank Fusion (RRF) scores across named vector fields.

        Args:
            candidate_ranks: Dict mapping candidate_id -> {vector_name: rank_index}.
            weights: Dict mapping vector_name -> float weight.
            k: Smoothing constant (default config.RRF_K_CONSTANT).

        Returns:
            Sorted list of tuples (candidate_id, rrf_score) descending.
        """
        k_val = k if k is not None else config.RRF_K_CONSTANT
        final_scores: dict[int, float] = {}

        for candidate_id, vector_ranks in candidate_ranks.items():
            rrf_sum = 0.0
            for vec_name, rank_idx in vector_ranks.items():
                w = weights.get(vec_name, 0.1)
                rrf_sum += w * (1.0 / (k_val + rank_idx + 1))
            final_scores[candidate_id] = rrf_sum

        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

hybrid_retriever = HybridRetrievalEngine()
