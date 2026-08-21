import time
import logging
import threading
from typing import Any, Dict, List, Optional
from db.neo4j_client import neo4j_client
from logger.unified_logger import get_logger, Subsystem

logger = get_logger(Subsystem.DB_REGISTRY)

class GraphRegistryCache:
    """Thread-safe TTL cache providing dynamic metadata extraction from Neo4j Graph DB."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._last_refresh: float = 0.0

        # Cached structures
        self._movie_titles: List[str] = []
        self._subgenre_map: Dict[str, str] = {}
        self._genre_alias_map: Dict[str, List[str]] = {}
        self._setting_keyword_map: Dict[str, List[str]] = {}

    def _should_refresh(self) -> bool:
        return (time.time() - self._last_refresh) > self.ttl_seconds or not self._movie_titles

    def refresh(self, force: bool = False) -> None:
        """Fetch latest metadata from Neo4j Graph DB."""
        with self._lock:
            if not force and not self._should_refresh():
                return

            logger.info("[GRAPH_REGISTRY] Refreshing dynamic graph metadata from Neo4j...")
            try:
                # 1. Fetch all Movie Titles dynamically
                title_res = neo4j_client.execute_query("MATCH (m:Movie) RETURN DISTINCT m.title AS title")
                self._movie_titles = [r["title"] for r in title_res if r.get("title")]

                # 2. Fetch all Subgenre Nodes dynamically
                subg_res = neo4j_client.execute_query("MATCH (s:Subgenre) RETURN DISTINCT s.name AS subgenre")
                sg_map: Dict[str, str] = {}
                for r in subg_res:
                    sg_name = r.get("subgenre")
                    if sg_name:
                        sg_map[sg_name.lower()] = sg_name
                        sg_map[sg_name.lower().replace("-", " ")] = sg_name
                        sg_map[sg_name.lower().replace(" ", "-")] = sg_map.get(sg_name.lower(), sg_name)
                
                # Add standard alias helpers for common user shorthand
                sg_map["romcom"] = "Romantic Comedy"
                sg_map["rom-com"] = "Romantic Comedy"
                sg_map["dystopia"] = "Dystopian"
                sg_map["psych horror"] = "Psychological Horror"
                self._subgenre_map = sg_map

                # 3. Fetch all Genre & Subgenre Names to build dynamic GENRE_ALIAS_MAP
                genre_res = neo4j_client.execute_query("MATCH (g:Genre) RETURN DISTINCT g.name AS genre")
                alias_map: Dict[str, List[str]] = {
                    "sci-fi": ["Science Fiction", "Sci-Fi", "SciFi"],
                    "science fiction": ["Science Fiction", "Sci-Fi", "SciFi"],
                    "scifi": ["Science Fiction", "Sci-Fi", "SciFi"],
                    "romcom": ["Romance", "Romantic Comedy"],
                    "rom-com": ["Romance", "Romantic Comedy"],
                    "cyberpunk": ["Cyberpunk", "Science Fiction"],
                    "dystopian": ["Dystopian", "Science Fiction"],
                    "psychological horror": ["Psychological Horror", "Horror"],
                    "slasher": ["Slasher", "Horror"],
                    "creature horror": ["Creature Horror", "Horror"],
                }

                for r in genre_res:
                    g_name = r.get("genre")
                    if g_name:
                        g_lower = g_name.lower()
                        if g_lower not in alias_map:
                            alias_map[g_lower] = [g_name]

                for sg_lower, canonical in sg_map.items():
                    if sg_lower not in alias_map:
                        alias_map[sg_lower] = [canonical]

                self._genre_alias_map = alias_map

                # 4. Fetch all Setting Nodes dynamically
                setting_res = neo4j_client.execute_query("MATCH (st:Setting) RETURN DISTINCT st.name AS setting")
                setting_map: Dict[str, List[str]] = {}
                for r in setting_res:
                    st_name = r.get("setting")
                    if st_name:
                        st_lower = st_name.lower()
                        setting_map[st_lower] = [st_name]
                        # Expand setting tokens (e.g. isolated_ship -> isolated, ship)
                        tokens = st_lower.replace("_", " ").split()
                        for token in tokens:
                            if len(token) > 3:
                                setting_map.setdefault(token, []).append(st_name)

                # Standard spatial setting keyword defaults
                setting_map.setdefault("space", ["space", "spaceship", "isolated_ship", "exoplanet", "wormhole"])
                setting_map.setdefault("spaceship", ["space", "spaceship", "isolated_ship"])
                setting_map.setdefault("motel", ["motel", "house"])
                setting_map.setdefault("antarctica", ["antarctica", "isolated_station"])

                self._setting_keyword_map = setting_map
                self._last_refresh = time.time()
                logger.info(f"[GRAPH_REGISTRY] Loaded {len(self._movie_titles)} movies, {len(self._subgenre_map)} subgenres, {len(self._setting_keyword_map)} settings dynamically.")

            except Exception as e:
                logger.warning(f"[GRAPH_REGISTRY] Graph metadata refresh warning (using existing cache): {e}")

    def get_known_movie_titles(self) -> List[str]:
        """Return list of all movie titles dynamically fetched from Neo4j."""
        with self._lock:
            self.refresh()
            return list(self._movie_titles)

    def get_subgenre_map(self) -> Dict[str, str]:
        """Return dynamic mapping of lower-case subgenre keywords to canonical Graph DB names."""
        with self._lock:
            self.refresh()
            return dict(self._subgenre_map)

    def get_genre_alias_map(self) -> Dict[str, List[str]]:
        """Return dynamic genre alias mapping table."""
        with self._lock:
            self.refresh()
            return dict(self._genre_alias_map)

    def get_setting_keyword_map(self) -> Dict[str, List[str]]:
        """Return dynamic spatial setting keyword mapping table."""
        with self._lock:
            self.refresh()
            return dict(self._setting_keyword_map)

# Singleton global registry instance
graph_registry = GraphRegistryCache()
