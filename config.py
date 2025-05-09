import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env
load_dotenv()

# OpenAI API credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default model (override via .env if needed)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Validate key presence
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found. Make sure it's set in your .env file or environment variables."
    )
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

DATA_PATH    = os.getenv("DATA_PATH", "./data/raw")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
