"""Qdrant client wrapper for managing vector database connections."""

from typing import Optional
from qdrant_client import QdrantClient
from config import config
from logger.unified_logger import get_logger, Subsystem

logger = get_logger(Subsystem.DB_QDRANT)

class QdrantClientWrapper:
    """Wrapper around QdrantClient for multi-vector search database operations."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Initialize Qdrant client connection parameters.

        Args:
            host: Qdrant host address.
            port: Qdrant REST API port.
        """
        self.host = host or config.QDRANT_HOST
        self.port = port or config.QDRANT_PORT
        self._client: Optional[QdrantClient] = None

    def get_client(self) -> QdrantClient:
        """Get or initialize active QdrantClient instance.

        Returns:
            QdrantClient instance.
        """
        if not self._client:
            self._client = QdrantClient(host=self.host, port=self.port, timeout=60.0)
        return self._client

qdrant_wrapper = QdrantClientWrapper()
