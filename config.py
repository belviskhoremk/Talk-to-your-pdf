"""Application settings, read from environment variables (or a ``.env`` file).

Copy ``.env.example`` to ``.env`` and fill in your own values. Secrets must
never be committed to the repository.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Hugging Face access token: https://huggingface.co/settings/tokens
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Flask signs the session cookie with this key. Use a long random string.
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")

# Models
LLM_REPO_ID = os.getenv("LLM_REPO_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Where uploaded PDFs are stored.
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")

# Text-splitting parameters: chunks of ~1500 characters that overlap by 150,
# so a sentence cut at a chunk boundary still appears whole in one chunk.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Older turns are dropped once the chat history exceeds this many words,
# so the prompt stays within the model's context window.
MAX_HISTORY_WORDS = 1024
