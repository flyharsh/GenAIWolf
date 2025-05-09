from typing import List, Dict
from core.interfaces import IRetriever, IVectorStore, IEmbedder
from .strategies import select_k

class Retriever(IRetriever):
    def __init__(self, embedder: IEmbedder, store: IVectorStore, default_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.default_k = default_k

    def retrieve(self, query: str) -> List[Dict]:
        # dynamically choose k
        k = select_k(query)
        vec = self.embedder.embed_query(query)
        return self.store.query(vec, k)
