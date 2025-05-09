# ui/main.py
import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="GenAIwolf", layout="wide")
st.title("🦾 GenAIwolf RAG Demo")

# 1) PDF upload
st.header("1. Upload Document")
uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    resp = requests.post(f"{BACKEND_URL}/upload", files={"file": uploaded_file})
    st.success(f"Upload status: {resp.json().get('status')}")

st.markdown("---")

# 2) Query interface
st.header("2. Ask a Question")
query = st.text_input("Enter your question", "")
if st.button("Ask GenAIwolf") and query:
    # Check cache bypass
    payload = {"query": query}
    resp = requests.post(f"{BACKEND_URL}/query", json=payload)
    data = resp.json()
    answer = data.get("answer", "No answer returned.")
    sources = data.get("sources", [])

    st.subheader("🔎 Answer")
    st.write(answer)

    if sources:
        st.subheader("📑 Sources")
        for src in sources:
            st.markdown(f"- **{src['source']}**: {src['text'][:200]}…")
