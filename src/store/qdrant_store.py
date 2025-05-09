import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from core.interfaces import IVectorStore

class QdrantStore(IVectorStore):
    def __init__(self, url: str, api_key: str, collection: str, dim: int):
        self.client = QdrantClient(url=url, api_key=api_key)
        # recreate or verify collection
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance="Cosine")
        )
        self.collection = collection

    def upsert(self, ids: List[str], vectors: List[List[float]], metadata: List[Dict]) -> None:
        points = []
        for _id, vec, meta in zip(ids, vectors, metadata):
            points.append(PointStruct(id=_id, vector=vec, payload=meta))
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, vector: List[float], k: int) -> List[Dict]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=k,
            with_payload=True
        )
        return [
            {"id": pt.id, **pt.payload}
            for pt in hits
        ]
