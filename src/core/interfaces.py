# src/core/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class IIngestor(ABC):
    @abstractmethod
    def ingest(self, source_path: str) -> List[Dict]:
        """Load & chunk a document into metadata dicts."""

class IEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text chunks into vectors."""
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string into a vector."""

class IVectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[List[float]], metadata: List[Dict]) -> None:
        """Upsert embeddings + metadata."""
    @abstractmethod
    def query(self, vector: List[float], k: int) -> List[Dict]:
        """Return top‑k items with payload for a query vector."""

class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> List[Dict]:
        """Return the top‑k text chunks (with metadata) for a natural‑language query."""

class ILLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Return model output for the given prompt."""

class IChain(ABC):
    @abstractmethod
    def run(self, query: str) -> Tuple[str, List[Dict]]:
        """End‑to‑end: retrieve context, call LLM, return answer and supporting chunks."""
