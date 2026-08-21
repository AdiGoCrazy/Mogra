"""Qdrant multi-vector ingestion module creating named vector points."""

import logging
from typing import Optional
from qdrant_client.models import PointStruct
from db.qdrant_client import QdrantClientWrapper, qdrant_wrapper
from schemas.enrichment import MovieEnrichmentPayload
from config import config

logger = logging.getLogger(__name__)

def ingest_movie_to_qdrant(
    payload: MovieEnrichmentPayload,
    vector_dict: dict[str, list[float]],
    wrapper: Optional[QdrantClientWrapper] = None
) -> None:
    """Ingest a movie's 6 named vectors and metadata payload into Qdrant.

    Args:
        payload: MovieEnrichmentPayload containing metadata attributes.
        vector_dict: Dict mapping vector_name (e.g. 'visual_aesthetic') -> float list embedding.
        wrapper: Qdrant client wrapper instance.
    """
    wrapper = wrapper or qdrant_wrapper
    client = wrapper.get_client()

    point = PointStruct(
        id=payload.tmdb_id,
        vector=vector_dict,
        payload={
            "tmdb_id": payload.tmdb_id,
            "title": payload.title,
            "release_year": payload.release_year,
            "synopsis": getattr(payload, "synopsis", getattr(payload.vector_payloads, "narrative_synopsis", payload.title)),
            "mpaa_rating": payload.ratings.mpaa_rating,
            "imdb_rating": payload.ratings.imdb_rating,
            "primary_genre": payload.taxonomy.primary_genre,
            "subgenres": payload.taxonomy.subgenres,
            "gore_level": payload.content_and_romance.gore_level,
            "has_romance": payload.negative_flags.has_romance,
        }
    )

    client.upsert(
        collection_name=config.QDRANT_COLLECTION,
        points=[point]
    )
    logger.info(f"Successfully upserted movie '{payload.title}' (ID: {payload.tmdb_id}) into Qdrant.")
