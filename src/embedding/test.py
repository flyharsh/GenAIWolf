# from openai import OpenAI
# from config import OPENAI_API_KEY
# client = OpenAI(api_key=OPENAI_API_KEY)
# print(OPENAI_API_KEY)

# response = client.responses.create(
#     model="gpt-3.5-turbo",
#     input="say hi."
# )

# print(response.output_text)
##################################################

# from qdrant_client import QdrantClient
# import config

# client = QdrantClient(
#     url=config.QDRANT_URL,
#     api_key=config.QDRANT_API_KEY,
#     prefer_grpc=False
# )
# print("Collections:", client.get_collections())
########################

# retrieve_from_qdrant.py

# retrieve_from_qdrant.py

# src/retrieve_from_qdrant.py

# src/embedding/test.py

# src/retrieve_manual_qdrant.py

# src/qa_qdrant_openai.py

# src/qa_qdrant_openai.py

import os
import uuid
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

import config

def main():
    # ── 1) Load env & init clients ────────────────────────────────
    load_dotenv()  # loads .env into os.environ
    # Qdrant
    qdrant = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY
    )
    # Embedding model
    embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    # OpenAI v1 client
    oa = OpenAI(api_key=config.OPENAI_API_KEY)

    print("✅ Clients initialized. Ready for queries.\n")

    # ── 2) Interactive QA loop ────────────────────────────────────
    while True:
        query = input("🔎 Question (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            print("✌️  Goodbye.")
            break

        # 2a) Embed the user query
        q_vec = embedder.embed_query(query)

        # 2b) Retrieve top‑5 from Qdrant
        hits = qdrant.search(
            collection_name=config.QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=5,
            with_payload=True
        )
        if not hits:
            print("⚠️  No matches found in your docs.")
            continue

        # 2c) Build a single context string
        snippets = []
        for pt in hits:
            payload = pt.payload or {}
            text    = payload.get("text", "").strip()
            if text:
                src = payload.get("source", "unknown")
                snippets.append(f"[{src}] {text}")
        context = "\n\n---\n\n".join(snippets)

        # 2d) Call GPT‑3.5‑Turbo via new v1 client
        system_msg = (
            "You are a corporate‑grade assistant. "
            "Answer ONLY using the CONTEXT provided. "
            "If the answer isn’t in the context, reply “I don’t know.”"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
        ]
        resp = oa.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=300
        )
        answer = resp.choices[0].message.content.strip()

        # 2e) Print answer
        print("\n✨ Answer:\n", answer, "\n")

if __name__ == "__main__":
    main()
