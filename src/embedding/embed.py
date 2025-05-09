# ingest_folder_to_qdrant_manual.py

import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import config

def load_docs(path: str):
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return PyPDFLoader(path).load()
    if ext == "docx":
        return Docx2txtLoader(path).load()
    if ext in ("txt", "md"):
        return TextLoader(path, encoding="utf-8").load()
    return []

def main():
    # 1) Discover & chunk
    files = [
        f for f in os.listdir(config.DATA_PATH)
        if f.lower().endswith((".pdf", ".docx", ".txt", ".md"))
    ]
    if not files:
        print(f"No docs found in {config.DATA_PATH}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for fname in files:
        path = os.path.join(config.DATA_PATH, fname)
        docs = load_docs(path)
        for c in splitter.split_documents(docs):
            c.metadata["source"] = fname
            chunks.append(c)
    print(f"Chunked {len(chunks)} segments from {len(files)} files.")

    # 2) Embed synchronously
    embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    texts = [c.page_content for c in chunks]
    embeddings = embedder.embed_documents(texts)
    dim = len(embeddings[0])

    # 3) Connect & (re)create collection
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        prefer_grpc=False
    )
    # If you want to drop any existing data, uncomment:
    # client.delete_collection(config.QDRANT_COLLECTION)
    client.recreate_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance="Cosine")
    )

    # 4) Upsert points
    points = []
    for i, (emb, chunk) in enumerate(zip(embeddings, chunks)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "source": chunk.metadata["source"],
                    "text": chunk.page_content
                }
            )
        )
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    print("✅ Ingest complete:", client.get_collections())

if __name__ == "__main__":
    main()
