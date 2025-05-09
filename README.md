# 🐺 GenAIwolf

GenAIwolf is a modular, scalable, and cost-efficient Retrieval-Augmented Generation (RAG) application. It enables users to ingest documents (PDF, DOCX, TXT), embed them using HuggingFace embeddings, store them in a Qdrant vector store, and retrieve context-based answers powered by OpenAI GPT-3.5 Turbo, optimizing LLM usage through caching and dynamic retrieval sizing.

---

## 🚀 Features

- **Document Ingestion**
  - PDF, DOCX, TXT file support
  - Smart chunking with overlapping texts
  - UUID-based chunk identification

- **Embedding & Vector Storage**
  - HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
  - Cloud-hosted Qdrant vector store integration

- **Retrieval & QA**
  - Dynamic context retrieval based on query complexity
  - Retrieval-QA chaining for grounded responses

- **Cost Optimization**
  - Redis-based caching for minimizing redundant OpenAI calls
  - Dynamic retrieval sizes to reduce unnecessary tokens usage

- **Containerization & Development**
  - Docker & Docker Compose for rapid development and deployment
  - Adheres to the Twelve-Factor App methodology (config via `.env`)

---

## 📁 Project Structure

```text
GenAIwolf/
├── .env                      # Environment variables and secrets
├── config.py                 # Loads configuration from `.env`
├── requirements.txt          # Python dependencies
├── Dockerfile.backend        # Backend Dockerfile (FastAPI)
├── Dockerfile.ui             # Frontend Dockerfile (Streamlit)
├── docker-compose.yml        # Docker Compose for service orchestration
├── data/                     
│   └── raw/                  # Raw documents (PDF/DOCX/TXT)
└── src/                      
    ├── api/                  # FastAPI backend
    │   ├── app.py            # Entry-point for FastAPI
    │   └── routers/          # Additional FastAPI routers
    ├── core/                 
    │   └── interfaces.py     # Abstract base classes/interfaces
    ├── ingest/               
    │   └── pdf_ingestor.py   # Document ingestion logic
    ├── embed/                
    │   └── hf_embedder.py    # Embedding logic using HuggingFace
    ├── store/                
    │   └── qdrant_store.py   # Qdrant storage adapter
    ├── retrieve/             
    │   ├── retriever.py      # Retriever implementation
    │   └── strategies.py     # Retrieval strategy implementation
    ├── llm/                  
    │   └── openai_llm.py     # OpenAI API client wrapper
    └── chain/                
        └── retrieval_qa.py   # Retrieval-QA chaining logic
