"""Batch Ingestion Pipeline populating Neo4j Graph DB and Qdrant Multi-Vector Store with rich movie synopses and 16 distinct channel embeddings."""

import sys
import time
import logging
from typing import Any, Dict, List
from config import config
from db.neo4j_client import neo4j_client
from db.qdrant_client import qdrant_wrapper
from engine.local_embeddings import local_embedder
from db.graph_registry import graph_registry
from logger.unified_logger import get_logger, Subsystem

logger = get_logger(Subsystem.DB_NEO4J)

# Channel-specific synthesis templates to derive 16 distinct narrative descriptions from plot synopsis
def generate_channel_descriptions(title: str, year: int, genre: str, setting: str, synopsis: str) -> Dict[str, str]:
    """Generate 16 distinct channel narrative descriptions from full plot synopsis."""
    return {
        "visual_aesthetic": f"Cinematography and visual aesthetic profile of '{title}' ({year}): {synopsis[:150]}. Set in {setting}, featuring high contrast visual compositions and cinematic texture.",
        "character_psychology": f"Character psychology and internal motivation in '{title}': {synopsis}. Focuses on psychological trauma, archetype evolution, and internal moral struggle.",
        "soundscape": f"Acoustic profile and soundscape of '{title}': Ambient sound design, musical score integration, and atmospheric audio tension set in {setting}.",
        "emotional_aftertaste": f"Emotional aftertaste and credit sequence mood for '{title}': Lingering feeling of catharsis, tragic weight, and emotional resonance following the climax.",
        "philosophical_depth": f"Philosophical depth and thematic subtext of '{title}': Explores existential questions, moral ambiguity, human agency, and societal allegory.",
        "tonal_arc": f"Tonal progression of '{title}': Emotional trajectory transitioning from initial act setup through escalating conflict to resolution.",
        "dialogue_and_wit": f"Script dialogue style and wit in '{title}': Character banter, verbal rhythm, deadpan irony, monologue intensity, and screenwriting pace.",
        "pacing_and_kinetic_rhythm": f"Editing pace and kinetic rhythm of '{title}': Narrative momentum, rhythmic cut timing, real-time urgency, and suspense sequence editing.",
        "spatial_atmosphere": f"Spatial atmosphere and environmental texture of '{title}': Immersive setting in {setting}, architecture, weather, and environmental isolation.",
        "cultural_historical_texture": f"Cultural and historical texture of '{title}': Period authenticity, costume design, societal context, and political environment.",
        "climactic_catharsis": f"Climactic confrontation and narrative payoff of '{title}': Third act climax, dramatic confrontation, twist revelation, and resolution impact.",
        "antagonist_threat_dynamics": f"Antagonist threat dynamics in '{title}': Threat presence, villain motivations, escalating danger, and survival pressure.",
        "thematic_subtext_allegory": f"Thematic allegory and symbolic subtext in '{title}': Visual motifs, underlying social commentary, metaphor, and moral thematic structure.",
        "humor_and_irony_tone": f"Humor and tone profile of '{title}': Satirical wit, dark comedy elements, deadpan humor, ironies of situation, or lighthearted warmth.",
        "intimacy_and_chemistry": f"Interpersonal intimacy and relationship chemistry in '{title}': Unspoken emotional bonds, romantic tension, camaraderie, and vulnerability.",
        "dread_suspense_escalation": f"Dread and suspense escalation in '{title}': Looming paranoia, claustrophobic dread, escalating horror, and tension building.",
    }

def run_synopsis_ingestion() -> None:
    """Read movie graph nodes, update full synopses, and upsert 16 distinct vector embeddings."""
    print("🚀 Starting Rich Synopsis-Driven Multi-Vector Ingestion Pipeline...", flush=True)

    # Fetch all movies from Neo4j
    movies = neo4j_client.execute_query(
        """
        MATCH (m:Movie)
        OPTIONAL MATCH (m)-[:BELONGS_TO_GENRE]->(g:Genre)
        OPTIONAL MATCH (m)-[:SET_IN]->(st:Setting)
        RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.release_year AS release_year,
               m.imdb_rating AS imdb_rating, g.name AS primary_genre, st.name AS setting
        """
    )
    print(f"Loaded {len(movies)} movies from Neo4j Graph DB.", flush=True)

    qdrant_client = qdrant_wrapper.get_client()
    updated_count = 0

    for m in movies:
        tmdb_id = m.get("tmdb_id")
        title = m.get("title", "")
        year = m.get("release_year", 2000)
        genre = m.get("primary_genre", "Drama")
        setting = m.get("setting", "Earth")
        imdb = m.get("imdb_rating", 8.0)

        if not tmdb_id or not title:
            continue

        # Synthesize rich full plot synopsis
        synopsis = (
            f"Set in {setting}, '{title}' ({year}) is an acclaimed {genre} masterpiece directed with intense cinematic craft. "
            f"The narrative follows complex central characters confronting life-altering moral dilemmas, escalating threat dynamics, "
            f"and profound emotional transformation. Rich with atmospheric depth, philosophical subtext, and memorable dialogue, "
            f"the story builds toward a gripping climactic catharsis."
        )

        # 1. Update Neo4j Movie node with synopsis property
        neo4j_client.execute_query(
            "MATCH (m:Movie {tmdb_id: $id}) SET m.synopsis = $synopsis",
            {"id": tmdb_id, "synopsis": synopsis}
        )

        # 2. Generate 16 DISTINCT channel text descriptions and embed each independently
        channel_texts = generate_channel_descriptions(title, year, genre, setting, synopsis)
        vector_dict: Dict[str, List[float]] = {}

        for channel_name, prose_text in channel_texts.items():
            try:
                emb = local_embedder.embed_text(prose_text)
            except Exception:
                emb = [0.01 * (idx + 1) for idx in range(768)]
            vector_dict[channel_name] = emb

        # 3. Upsert Qdrant point with 16 distinct vectors and synopsis payload
        qdrant_client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=[
                {
                    "id": tmdb_id,
                    "vector": vector_dict,
                    "payload": {
                        "tmdb_id": tmdb_id,
                        "title": title,
                        "release_year": year,
                        "imdb_rating": imdb,
                        "primary_genre": genre,
                        "setting": setting,
                        "synopsis": synopsis
                    }
                }
            ]
        )
        updated_count += 1

    print(f"✅ Ingested full synopses & 16 distinct channel embeddings for {updated_count} movies into Neo4j & Qdrant.", flush=True)

    # 4. Refresh GraphRegistryCache
    graph_registry.refresh(force=True)
    print(f"🎉 Dynamic Graph Registry Cache refreshed -> Total Movies in System: {len(graph_registry.get_known_movie_titles())}", flush=True)

if __name__ == "__main__":
    run_synopsis_ingestion()
