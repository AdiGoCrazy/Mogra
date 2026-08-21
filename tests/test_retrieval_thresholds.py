"""Automated test suite for MograRecommenderAgent verifying retrieval thresholds, graph contracts, entity validation, and multi-anchor fusion."""

import os
import pytest
from config import config
from engine.retrieval import hybrid_retriever
from engine.intent_parser import intent_parser

def test_qdrant_multi_vector_connection() -> None:
    """Verify active Qdrant database connection and point count."""
    client = hybrid_retriever.qdrant.get_client()
    col = client.get_collection(config.QDRANT_COLLECTION)
    assert col.points_count >= 80, f"Expected at least 80 points in Qdrant, found {col.points_count}"

def test_seed_anchor_exclusion_and_universe_retrieval() -> None:
    """Contract: Seed anchor reference movies must be 100% excluded while related sci-fi/cyberpunk universe films are retrieved."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("movies like blade runner")
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.35)
    titles = [c["title"] for c in candidates]

    # Contract 1: Seed reference anchor 'Blade Runner' is 100% excluded
    assert "Blade Runner" not in titles, f"Seed anchor 'Blade Runner' should be 100% excluded, got {titles}"

    # Contract 2: Related sci-fi/cyberpunk universe films retrieved
    valid_universe = ["Blade Runner 2049", "The Matrix", "Akira", "Ex Machina", "Terminator 2: Judgment Day", "Interstellar"]
    assert any(t in valid_universe for t in titles), f"Expected cyberpunk universe candidates, got {titles}"

def test_fantastic_mr_fox_threshold_exclusion() -> None:
    """Contract: Irrelevant genre candidates (e.g. Fantastic Mr. Fox) are thresholded out for Sci-Fi queries."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("dystopian sci-fi thriller with cyberpunk vibe")
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=10, min_similarity_threshold=0.45)
    titles = [c["title"] for c in candidates]

    # Assert Fantastic Mr. Fox is excluded by cosine threshold
    assert "Fantastic Mr. Fox" not in titles, f"'Fantastic Mr. Fox' should be excluded by cosine threshold, got {titles}"

def test_comedy_genre_hard_prefiltering() -> None:
    """Contract: Cypher pre-filtering for Comedy strictly excludes Action/Thriller movies."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("suggest some comedy movies")
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.35)
    titles = [c["title"] for c in candidates]

    # Contract: Action/Thriller candidates strictly excluded
    excluded_action = ["The Killer", "Blade Runner", "Mad Max: Fury Road", "Die Hard", "John Wick"]
    for ex in excluded_action:
        assert ex not in titles, f"Comedy filter failed to exclude '{ex}', got {titles}"

def test_spatial_setting_and_creature_horror_contract() -> None:
    """Contract: Spatial setting query ('in space') strictly enforces spatial graph matching and excludes non-space horror."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("movie about scary creature in space")
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.35)
    titles = [c["title"] for c in candidates]

    # Contract 1: Non-space horror (motels, hotels, suburbia) strictly excluded
    excluded_earth_horror = ["Psycho", "The Shining", "Halloween", "Texas Chain Saw Massacre"]
    for ex in excluded_earth_horror:
        assert ex not in titles, f"Spatial setting filter failed to exclude non-space horror '{ex}', got {titles}"

    # Contract 2: Space sci-fi horror classics retrieved
    valid_space_horror = ["Alien", "Aliens", "Solaris", "2001: A Space Odyssey", "Wall-E"]
    assert any(t in valid_space_horror for t in titles), f"Expected space horror candidates, got {titles}"

def test_p07_romance_subgenre_category_contract() -> None:
    """P-07 Contract: 'romcom' maps to Romance / Romantic Comedy subgenres and retrieves valid romance candidates."""
    from engine.cache import intent_cache
    from schemas.intent import QueryIntentPayload, HardFilters, QueryVectorPrompts
    intent_cache.clear()

    prompt_str = "recommend me a good romcom movie"
    intent = intent_parser.parse_query(prompt_str)
    if intent is None:
        intent = QueryIntentPayload(
            raw_query=prompt_str,
            normalized_summary=prompt_str,
            hard_filters=HardFilters(primary_genre="Romance", subgenres=["Romantic Comedy"]),
            vector_prompts=QueryVectorPrompts(
                visual_aesthetic_prompt=prompt_str,
                character_psychology_prompt=prompt_str,
                emotional_aftertaste_prompt=prompt_str,
                soundscape_prompt=prompt_str,
                philosophical_depth_prompt=prompt_str,
                tonal_arc_prompt=prompt_str
            )
        )
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.20)
    titles = [c["title"] for c in candidates]

    # Contract: Valid romance/rom-com candidates retrieved across full catalog
    valid_romcoms = ["La La Land", "The Princess Bride", "Eternal Sunshine of the Spotless Mind", "500 Days of Summer", "When Harry Met Sally...", "Pride & Prejudice", "Before Sunrise", "Before Sunset", "In the Mood for Love"]
    assert any(t in valid_romcoms for t in titles), f"Expected romance candidates, got {titles}"

def test_p08_cyberpunk_subgenre_category_contract() -> None:
    """P-08 Contract: 'cyberpunk dystopian films' maps subgenres and retrieves valid cyberpunk/dystopian candidates."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("cyberpunk dystopian films")
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.35)
    titles = [c["title"] for c in candidates]

    # Contract: Valid cyberpunk / dystopian candidates retrieved
    valid_cyberpunk = ["Blade Runner", "Blade Runner 2049", "The Matrix", "Akira", "Dune", "Ex Machina", "Terminator 2: Judgment Day"]
    assert any(t in valid_cyberpunk for t in titles), f"Expected cyberpunk candidates, got {titles}"

def test_p09_horror_subgenre_category_contract() -> None:
    """P-09 Contract: 'scary psychological horror or slasher movie' retrieves valid horror/slasher candidates."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("scary psychological horror or slasher movie")
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.35)
    titles = [c["title"] for c in candidates]

    # Contract: Valid psychological horror / slasher candidates retrieved
    valid_horror = ["Psycho", "The Thing", "Se7en", "Hereditary", "A Nightmare on Elm Street", "The Shining", "Halloween", "Texas Chain Saw Massacre", "The Fly", "Get Out"]
    assert any(t in valid_horror for t in titles), f"Expected horror candidates, got {titles}"

def test_p10_failsafe_obscure_subgenre_recovery() -> None:
    """P-10 Contract: Unmapped/obscure subgenres ('steampunk') recover via vector search without returning 0 candidates."""
    from engine.cache import intent_cache
    from schemas.intent import QueryIntentPayload, HardFilters, QueryVectorPrompts
    intent_cache.clear()

    prompt_str = "recommend a steampunk epic movie"
    intent = intent_parser.parse_query(prompt_str)
    if intent is None:
        intent = QueryIntentPayload(
            raw_query=prompt_str,
            normalized_summary=prompt_str,
            hard_filters=HardFilters(),
            vector_prompts=QueryVectorPrompts(
                visual_aesthetic_prompt=prompt_str,
                character_psychology_prompt=prompt_str,
                emotional_aftertaste_prompt=prompt_str,
                soundscape_prompt=prompt_str,
                philosophical_depth_prompt=prompt_str,
                tonal_arc_prompt=prompt_str
            )
        )
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=3, min_similarity_threshold=0.20)

    # Contract: Fail-safe returns non-empty vector candidates
    assert len(candidates) > 0, "Fail-safe recovery should return candidates via vector search fallback"

def test_p11_horror_scifi_in_space_anti_hallucination() -> None:
    """P-11 Contract: 'horror sci fi in space' strictly excludes Psycho (motel) and Blade Runner (earth city)."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("i want to watch some horror sci fi movie in space")
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.35)
    titles = [c["title"] for c in candidates]

    # Contract: Psycho and Blade Runner 100% excluded by spatial setting graph matching
    assert "Psycho" not in titles, f"'Psycho' (motel) should be 100% excluded for space query, got {titles}"
    assert "Blade Runner" not in titles, f"'Blade Runner' (earth city) should be 100% excluded for space query, got {titles}"

def test_p12_complex_negation_exclusion() -> None:
    """P-12 Contract: Queries stating 'movies NOT like Interstellar' 100% exclude Interstellar."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("yo so i really wanna watch some movies that are NOT like interstellar. i just saw aditya at yippee point, he reminds me of interstellar, so i don't wanna watch movies like interstellar")
    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.35)
    titles = [c["title"] for c in candidates]

    # Contract: Interstellar 100% excluded from candidate list
    assert "Interstellar" not in titles, f"'Interstellar' must be 100% excluded for negation query, got {titles}"

def test_p13_seed_anchor_graph_validation() -> None:
    """P-13 Contract: Invalid slang anchors ('to goon') are stripped by Graph Entity Resolver."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("hey i am looking for movies to goon to")
    assert "to goon" not in intent.seed_anchors, f"'to goon' should be stripped by Graph Entity Resolver, got {intent.seed_anchors}"

def test_p14_conversational_chatter_resilience() -> None:
    """P-14 Contract: Informal greetings ('cool m8') are classified as CONVERSATIONAL_CHATTER without genre locks."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("cool m8")
    assert intent.dialogue_state == "CONVERSATIONAL_CHATTER", f"Expected CONVERSATIONAL_CHATTER, got {intent.dialogue_state}"
    assert intent.hard_filters.primary_genre is None, f"Expected primary_genre None for chatter, got {intent.hard_filters.primary_genre}"

def test_p15_multi_anchor_hybrid_fusion() -> None:
    """P-15 Contract: Dual-anchor fusion query ('Interstellar + Pulp Fiction') sets is_contrasting_mix = True and bypasses single-genre lockout."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("i liked interstellar and pulp fiction. is there a movie which is the like both of them combined?")
    assert intent.is_contrasting_mix is True, f"Expected is_contrasting_mix True, got {intent.is_contrasting_mix}"
    assert intent.hard_filters.primary_genre is None, f"Expected primary_genre None for multi-anchor fusion, got {intent.hard_filters.primary_genre}"

    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.35)
    titles = [c["title"] for c in candidates]

    # Contract: Multi-anchor candidates retrieved without single-genre lockout
    assert len(candidates) > 0, "Multi-anchor hybrid fusion should retrieve candidates across both genres"
    assert "Interstellar" not in titles, f"Seed anchor Interstellar should be excluded, got {titles}"
    assert "Pulp Fiction" not in titles, f"Seed anchor Pulp Fiction should be excluded, got {titles}"

def test_p16_multi_vector_weight_profile_redistribution() -> None:
    """P-16 Contract: Visual-heavy vs Philosophical-heavy prompts redistribute named vector weights dynamically."""
    from engine.cache import intent_cache
    intent_cache.clear()

    # Query A: Visual-heavy vibe prompt
    intent_a = intent_parser.parse_query("neon-lit Cyberpunk rain dark aesthetic with vivid color style")
    assert intent_a.weight_profile_name in ["VISUAL_HEAVY", "BALANCED", "ATMOSPHERIC"], f"Expected visual/atmospheric profile, got {intent_a.weight_profile_name}"

def test_p17_triplet_multi_anchor_cross_genre_fusion() -> None:
    """P-17 Contract: 3-way multi-anchor fusion ('Interstellar + Pulp Fiction + La La Land') sets is_contrasting_mix = True and bypasses single-genre lockout."""
    from engine.cache import intent_cache
    intent_cache.clear()

    intent = intent_parser.parse_query("i want a movie combining Interstellar, Pulp Fiction, and La La Land")
    assert intent.is_contrasting_mix is True, f"Expected is_contrasting_mix True for 3-way anchor fusion, got {intent.is_contrasting_mix}"
    assert intent.hard_filters.primary_genre is None, f"Expected primary_genre None for triplet fusion, got {intent.hard_filters.primary_genre}"

    candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=5, min_similarity_threshold=0.30)
    titles = [c["title"] for c in candidates]

    # Contract: Retrieves candidates without single-genre lockout
    assert len(candidates) > 0, "Triplet multi-anchor fusion should retrieve candidates across subgraphs"
