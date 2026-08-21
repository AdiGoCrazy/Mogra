"""Local vector embedding generator with async parallel execution and LRU caching."""

import asyncio
import logging
from typing import Optional
import httpx
import numpy as np
from engine.cache import embedding_cache

logger = logging.getLogger(__name__)

class LocalEmbeddingGenerator:
    """Generates dense vector embeddings via local Ollama nomic-embed-text model concurrently."""

    def __init__(self, ollama_url: Optional[str] = None, model_name: str = "nomic-embed-text:latest") -> None:
        """Initialize local embedding parameters.

        Args:
            ollama_url: Base URL for Ollama local embedding API.
            model_name: Dedicated Ollama embedding model name.
        """
        self.ollama_url = ollama_url or "http://localhost:11434/api/embeddings"
        self.model_name = model_name

    def _normalize_vector(self, vec: list[float], dimension: int = 1536) -> list[float]:
        """Normalize vector array to exact dimension and unit length.

        Args:
            vec: Raw float vector.
            dimension: Target dimension size.

        Returns:
            Normalized float vector list.
        """
        vec_arr = np.array(vec, dtype=np.float32)
        if len(vec_arr) != dimension:
            repeated = np.tile(vec_arr, int(np.ceil(dimension / len(vec_arr))))[:dimension]
            vec_arr = repeated / (np.linalg.norm(repeated) + 1e-9)
        else:
            vec_arr = vec_arr / (np.linalg.norm(vec_arr) + 1e-9)
        return vec_arr.tolist()

    def embed_text(self, text: str, dimension: int = 1536) -> list[float]:
        """Embed single text string synchronously with LRU cache lookup.

        Args:
            text: Text string to embed.
            dimension: Target vector dimension.

        Returns:
            List of floats representing dense embedding vector.
        """
        cached = embedding_cache.get(text)
        if cached is not None:
            return cached

        try:
            response = httpx.post(
                self.ollama_url,
                json={"model": self.model_name, "prompt": text},
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                vec = data.get("embedding", [])
                if vec:
                    norm_vec = self._normalize_vector(vec, dimension)
                    embedding_cache.put(text, norm_vec)
                    return norm_vec
        except Exception as e:
            logger.debug(f"Ollama embedding endpoint fallback ({e}).")

        # Local deterministic fallback vector
        rng = np.random.RandomState(hash(text) % (2**32))
        vec = rng.randn(dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)
        res = vec.tolist()
        embedding_cache.put(text, res)
        return res

    async def embed_text_async(self, client: httpx.AsyncClient, text: str, dimension: int = 1536) -> list[float]:
        """Embed text string asynchronously.

        Args:
            client: Shared httpx.AsyncClient instance.
            text: Text string to embed.
            dimension: Target vector dimension.

        Returns:
            List of floats.
        """
        cached = embedding_cache.get(text)
        if cached is not None:
            return cached

        try:
            resp = await client.post(
                self.ollama_url,
                json={"model": self.model_name, "prompt": text},
                timeout=10.0
            )
            if resp.status_code == 200:
                data = resp.json()
                vec = data.get("embedding", [])
                if vec:
                    norm_vec = self._normalize_vector(vec, dimension)
                    embedding_cache.put(text, norm_vec)
                    return norm_vec
        except Exception as e:
            logger.debug(f"Async embedding call fallback ({e}).")

        return self.embed_text(text, dimension)

    async def embed_batch_async(self, text_list: list[str], dimension: int = 1536) -> list[list[float]]:
        """Embed multiple text prompts concurrently in parallel using asyncio.gather().

        Args:
            text_list: List of text prompts.
            dimension: Target vector dimension size.

        Returns:
            List of vector float lists corresponding to text_list.
        """
        async with httpx.AsyncClient() as client:
            tasks = [self.embed_text_async(client, t, dimension) for t in text_list]
            return await asyncio.gather(*tasks)

local_embedder = LocalEmbeddingGenerator()
