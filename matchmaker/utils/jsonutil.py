"""Parse JSON from LLM output (handles optional markdown fences)."""

import json
from typing import Any


def parse_json_loose(text: str) -> Any:
    """Strip ``` fences if present and parse JSON."""
    s = text.strip()
    if not s.startswith("```"):
        return json.loads(s)
    # Drop opening ``` or ```json
    first_nl = s.find("\n")
    if first_nl == -1:
        return json.loads(s)
    s = s[first_nl + 1 :]
    s = s.rstrip()
    if s.endswith("```"):
        s = s[: s.rfind("```")].rstrip()
    return json.loads(s)
