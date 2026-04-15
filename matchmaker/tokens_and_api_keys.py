import os

# Backward-compatible as I started hardcoding it in this file
open_ai_api_key: str = os.environ.get("OPENAI_API_KEY", "").strip()
