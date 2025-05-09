# src/api/app.py

from fastapi import FastAPI, UploadFile, File, HTTPException
import os, json
from dotenv import load_dotenv
from pydantic import BaseModel
import tempfile
import config

from ingest.pdf_ingestor import PDFIngestor
from embed.hf_embedder    import HFEmbedder
from store.qdrant_store   import QdrantStore
from retrieve.retriever   import Retriever
from llm.openai_llm       import OpenAIClient
from chain.retrieval_qa   import RetrievalQAChain


# Load environment variables from .env at startup
load_dotenv()

# Instantiate the FastAPI application
app = FastAPI()

# ─────────────────────────────────────────────────────────────────────────────
# Module initializations
# ─────────────────────────────────────────────────────────────────────────────

# Create a PDFIngestor to load & chunk documents
ingestor = PDFIngestor()

# Create an HFEmbedder using the EMBEDDING_MODEL env var
embedder = HFEmbedder(os.getenv("EMBEDDING_MODEL"))

# Determine embedding dimension by embedding a dummy text
# This dimension is needed to configure the Qdrant vector store
sample_vec = embedder.embed_query("hello")
dim = len(sample_vec)

# Initialize the QdrantStore with URL, API key, collection name, and vector dim
store = QdrantStore(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection=os.getenv("QDRANT_COLLECTION"),
    dim=dim
)

# Create a Retriever that ties together the embedder and vector store
retriever = Retriever(embedder, store)

# Instantiate the OpenAI client for LLM calls using API key and model name
llm_client = OpenAIClient(
    api_key=config.OPENAI_API_KEY,
    model=config.OPENAI_MODEL
)

# Build the RetrievalQAChain by combining retriever and LLM client
qa_chain = RetrievalQAChain(retriever, llm_client)

# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Accepts a file upload, chunks, embeds, and upserts into Qdrant.
    """
    # 1) Determine a safe temp dir
    #    Use system temp, or fall back to ./tmp in your project
    base_tmp = tempfile.gettempdir()
    # Optional: or use project-local tmp
    # base_tmp = os.path.join(os.getcwd(), "tmp")
    # os.makedirs(base_tmp, exist_ok=True)

    # 2) Build the full path
    path = os.path.join(base_tmp, file.filename)

    # 3) Write the file bytes to disk
    with open(path, "wb") as f:
        f.write(await file.read())

    # 4) Ingest, embed, and upsert as before
    chunks   = ingestor.ingest(path)
    texts    = [c["text"] for c in chunks]
    vectors  = embedder.embed(texts)
    ids      = [c["id"]   for c in chunks]
    metadata = [{"source": c["source"], "text": c["text"]} for c in chunks]
    store.upsert(ids, vectors, metadata)

    return {"status": "ingested", "count": len(chunks)}

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

@app.post("/query")
def query(body: QueryRequest):
    # FastAPI will automatically return 422 if "query" is missing or wrong type
    answer, sources = qa_chain.run(body.query)
    return {"answer": answer, "sources": sources}