"""Unit tests for dynamic GraphRegistryCache service."""

import pytest
from db.graph_registry import graph_registry, GraphRegistryCache

def test_graph_registry_dynamic_titles() -> None:
    """Verify GraphRegistryCache dynamically queries Neo4j for movie titles."""
    titles = graph_registry.get_known_movie_titles()
    assert isinstance(titles, list)
    assert len(titles) > 0
    assert any("blade runner" in t.lower() or "alien" in t.lower() for t in titles)

def test_graph_registry_dynamic_subgenres_and_settings() -> None:
    """Verify GraphRegistryCache dynamically queries Neo4j for subgenres and settings."""
    subg_map = graph_registry.get_subgenre_map()
    assert isinstance(subg_map, dict)
    assert len(subg_map) > 0
    assert "cyberpunk" in subg_map or "slasher" in subg_map

    setting_map = graph_registry.get_setting_keyword_map()
    assert isinstance(setting_map, dict)
    assert len(setting_map) > 0
    assert "space" in setting_map or "motel" in setting_map

def test_graph_registry_refresh_force() -> None:
    """Verify GraphRegistryCache force refresh updates cached structures safely."""
    cache = GraphRegistryCache(ttl_seconds=1.0)
    cache.refresh(force=True)
    t1 = cache._last_refresh
    assert t1 > 0

    titles = cache.get_known_movie_titles()
    assert len(titles) > 0
