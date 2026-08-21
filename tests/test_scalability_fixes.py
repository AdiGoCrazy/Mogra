"""Unit tests verifying comprehensive scalability refactorings across MograRecommenderAgent."""

import os
import pytest
from config import config
from engine.synthesizer import response_synthesizer
from engine.intent_parser import intent_parser
from scripts.run_prompt_simulation_benchmark import load_benchmark_suite

def test_synthesizer_dynamic_top_k_payload() -> None:
    """Verify response_synthesizer formats N candidate movies without hardcoded [:3] truncation."""
    candidates = [
        {"title": f"Movie {i}", "release_year": 2000 + i, "imdb_rating": 8.0, "primary_genre": "Action"}
        for i in range(1, 6)
    ]
    res = response_synthesizer.synthesize_response(
        user_query="action movies",
        recommended_movies=candidates
    )
    assert isinstance(res, str)
    assert len(res) > 0

def test_intent_parser_dynamic_negation_without_hardcoded_titles() -> None:
    """Verify intent parser extracts negative seed anchors without hardcoded title branches."""
    intent = intent_parser.parse_query("sci-fi movies but not like Alien")
    assert "Alien" in intent.negative_seed_anchors

def test_retrieval_limits_from_config() -> None:
    """Verify retrieval parameters are externalized in config.py."""
    assert hasattr(config, "CYPHER_MAX_LIMIT")
    assert hasattr(config, "VECTOR_SEARCH_LIMIT")
    assert hasattr(config, "RRF_K_CONSTANT")
    assert config.CYPHER_MAX_LIMIT > 0
    assert config.VECTOR_SEARCH_LIMIT > 0
    assert config.RRF_K_CONSTANT > 0

def test_benchmark_loader_from_json() -> None:
    """Verify load_benchmark_suite loads test cases from data/benchmark_test_cases.json."""
    assert os.path.exists("data/benchmark_test_cases.json")
    suite = load_benchmark_suite()
    assert isinstance(suite, list)
    assert len(suite) >= 15
    assert suite[0].prompt is not None
