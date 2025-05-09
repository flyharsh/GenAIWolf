# src/ingest/pdf_ingestor.py

import os
import uuid
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.interfaces import IIngestor

class PDFIngestor(IIngestor):
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def ingest(self, source_path: str) -> List[Dict]:
        # Load the document based on file extension
        ext = os.path.splitext(source_path)[1].lower()
        if ext == ".pdf":
            docs = PyPDFLoader(source_path).load()
        elif ext == ".docx":
            docs = Docx2txtLoader(source_path).load()
        else:
            docs = TextLoader(source_path, encoding="utf-8").load()

        # Split into chunks
        chunks = self.splitter.split_documents(docs)
        result = []
        for c in chunks:
            text = c.page_content or ""
            if not text.strip():
                continue

            # Generate a UUID4 for each chunk—valid for Qdrant IDs
            chunk_id = str(uuid.uuid4())

            result.append({
                "id": chunk_id,
                "text": text,
                "source": os.path.basename(source_path)
            })
        return result
