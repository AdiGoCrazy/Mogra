"""Ingestion runner loading seed benchmark movies into Neo4j & Qdrant with zero API keys required."""

import json
import logging
from pathlib import Path
from schemas.enrichment import MovieEnrichmentPayload
from ingestion.neo4j_ingestor import ingest_movie_to_neo4j
from ingestion.qdrant_ingestor import ingest_movie_to_qdrant
from engine.local_embeddings import local_embedder
from db.neo4j_client import neo4j_client
from db.neo4j_schema import initialize_neo4j_schema
from db.qdrant_client import qdrant_wrapper
from db.qdrant_schema import initialize_qdrant_schema
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion_runner")

def run_seed_ingestion(json_path: str = "data/seed_movies.json") -> int:
    """Load benchmark seed movies, embed vectors locally, and ingest into Neo4j & Qdrant.

    Args:
        json_path: Path to seed movies JSON file.

    Returns:
        Total number of movies successfully ingested.
    """
    logger.info("--- STARTING LOCAL MOVIE INGESTION (Zero API Keys Required) ---")

    # 1. Initialize Database Schemas
    logger.info("Initializing Neo4j Graph constraints & Qdrant multi-vector collection...")
    try:
        initialize_neo4j_schema(neo4j_client)
        initialize_qdrant_schema(qdrant_wrapper)
    except Exception as e:
        logger.warning(f"Database schema initialization warning: {e}. (Ensure Docker containers are running).")

    # 2. Read seed movies dataset
    file_path = Path(json_path)
    if not file_path.exists():
        logger.error(f"Seed movies file not found at '{json_path}'.")
        return 0

    with open(file_path, "r", encoding="utf-8") as f:
        movies_raw = json.load(f)

    logger.info(f"Loaded {len(movies_raw)} benchmark movies from '{json_path}'.")

    success_count = 0
    vector_names = [
        "visual_aesthetic",
        "character_psychology",
        "emotional_aftertaste",
        "soundscape",
        "philosophical_depth",
        "tonal_arc",
    ]

    for movie_dict in movies_raw:
        try:
            # Validate payload schema
            payload = MovieEnrichmentPayload.model_validate(movie_dict)
            logger.info(f"Processing '{payload.title}' (ID: {payload.tmdb_id})...")

            # 3. Generate 6 Named Dense Vector Embeddings Locally
            vector_dict: dict[str, list[float]] = {}
            p = payload.vector_payloads
            text_map = {
                "visual_aesthetic": p.visual_aesthetic_description,
                "character_psychology": p.character_psychology_description,
                "emotional_aftertaste": p.emotional_aftertaste_description,
                "soundscape": p.soundscape_description,
                "philosophical_depth": p.philosophical_depth_description,
                "tonal_arc": p.tonal_arc_description,
            }

            for vec_name in vector_names:
                text_to_embed = text_map[vec_name]
                vector_dict[vec_name] = local_embedder.embed_text(text_to_embed, dimension=config.VECTOR_SIZE)

            # 4. Ingest into Neo4j Graph DB
            try:
                ingest_movie_to_neo4j(payload, neo4j_client)
            except Exception as ne:
                logger.warning(f"Neo4j ingestion error for '{payload.title}': {ne}")

            # 5. Ingest into Qdrant Multi-Vector DB
            try:
                ingest_movie_to_qdrant(payload, vector_dict, qdrant_wrapper)
            except Exception as qe:
                logger.warning(f"Qdrant ingestion error for '{payload.title}': {qe}")

            success_count += 1

        except Exception as e:
            logger.error(f"Error processing movie payload: {e}")

    logger.info(f"--- INGESTION COMPLETE: {success_count}/{len(movies_raw)} Movies Successfully Ingested ---")
    return success_count

if __name__ == "__main__":
    run_seed_ingestion()
