from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from core.interfaces import IEmbedder

class HFEmbedder(IEmbedder):
    def __init__(self, model_name: str):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # embed_documents returns List[List[float]]
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_query(query)
