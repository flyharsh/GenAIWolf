from typing import List, Dict
from core.interfaces import ILLM, IRetriever

class RetrievalQAChain:
    def __init__(self, retriever: IRetriever, llm: ILLM):
        self.retriever = retriever
        self.llm = llm

    def run(self, query: str) -> (str, List[Dict]):
        chunks = self.retriever.retrieve(query)
        context = "\n\n---\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
        prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
        answer = self.llm.generate(prompt)
        return answer, chunks
