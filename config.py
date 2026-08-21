import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Neo4j Graph DB
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

    # Qdrant Multi-Vector DB
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "movies_multi_vector")
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))

    # Local LLM (Ollama)
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b")
    LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "ollama")

    # Retrieval Scaling Parameters
    CYPHER_MAX_LIMIT = int(os.getenv("CYPHER_MAX_LIMIT", "500"))
    VECTOR_SEARCH_LIMIT = int(os.getenv("VECTOR_SEARCH_LIMIT", "50"))
    RRF_K_CONSTANT = int(os.getenv("RRF_K_CONSTANT", "60"))

config = Config()
