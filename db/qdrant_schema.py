"""Qdrant schema initialization establishing named multi-vectors collection."""

import logging
from qdrant_client.models import VectorParams, Distance
from db.qdrant_client import QdrantClientWrapper
from config import config

logger = logging.getLogger(__name__)

NAMED_VECTORS: list[str] = [
    "visual_aesthetic",
    "character_psychology",
    "emotional_aftertaste",
    "soundscape",
    "philosophical_depth",
    "tonal_arc",
    "dialogue_and_wit",
    "pacing_and_kinetic_rhythm",
    "spatial_atmosphere",
    "cultural_historical_texture",
    "climactic_catharsis",
    "antagonist_threat_dynamics",
    "thematic_subtext_allegory",
    "humor_and_irony_tone",
    "intimacy_and_chemistry",
    "dread_suspense_escalation",
]

def initialize_qdrant_schema(wrapper: QdrantClientWrapper) -> None:
    """Initialize Qdrant collection with 6 named vector configurations.

    Args:
        wrapper: QdrantClientWrapper instance.
    """
    client = wrapper.get_client()
    collection_name = config.QDRANT_COLLECTION
    vector_size = config.VECTOR_SIZE

    logger.info(f"Initializing Qdrant Collection: {collection_name} (Size: {vector_size})...")

    # Build multi-vector parameters for all 6 named vectors
    vectors_config = {
        vector_name: VectorParams(size=vector_size, distance=Distance.COSINE)
        for vector_name in NAMED_VECTORS
    }

    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )
        logger.info(f"Successfully created Qdrant collection '{collection_name}' with 6 named vectors.")
    else:
        logger.info(f"Qdrant collection '{collection_name}' already exists.")
