"""
Persistent semantic memory using ChromaDB + all-MiniLM-L6-v2 embeddings.
Stores research results and retrieves relevant context before web search.
"""

import os
import uuid
import time
import logging
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
THRESHOLD          = float(os.getenv("MEMORY_SEARCH_THRESHOLD", "0.75"))
MAX_RESULTS        = int(os.getenv("MAX_MEMORY_RESULTS", "5"))


class ResearchMemory:
    """Semantic memory store backed by ChromaDB with local sentence-transformer embeddings."""

    COLLECTION = "research_memory"

    def __init__(self):
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ResearchMemory ready — %d documents loaded", self._col.count())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = MAX_RESULTS) -> list[dict]:
        """
        Return stored memories whose cosine similarity to *query* exceeds THRESHOLD.
        Each result dict has keys: id, query, content, source, timestamp, similarity.
        """
        if self._col.count() == 0:
            return []

        embedding = self._embed(query)
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist          # cosine distance → similarity
            if similarity >= THRESHOLD:
                hits.append(
                    {
                        "id": meta.get("doc_id", ""),
                        "query": meta.get("original_query", ""),
                        "content": doc,
                        "source": meta.get("source", "memory"),
                        "timestamp": meta.get("timestamp", ""),
                        "similarity": round(similarity, 4),
                    }
                )
        hits.sort(key=lambda x: x["similarity"], reverse=True)
        return hits

    def store(self, query: str, content: str, source: str = "web_search") -> str:
        """Embed and persist a research result. Returns the assigned doc_id."""
        doc_id = str(uuid.uuid4())
        embedding = self._embed(content)

        self._col.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "original_query": query,
                    "source": source,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ],
        )
        logger.info("Stored memory %s for query: %s", doc_id, query[:60])
        return doc_id

    def count(self) -> int:
        return self._col.count()

    def clear(self):
        """Delete all documents (useful for testing)."""
        self._client.delete_collection(self.COLLECTION)
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Memory cleared.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()
