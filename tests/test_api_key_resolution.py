from pathlib import Path

from matchmaker.extractors.extract import _resolve_api_key


def test_resolve_api_key_none() -> None:
    assert _resolve_api_key(None) is None


def test_resolve_api_key_raw_string() -> None:
    assert _resolve_api_key("sk-test-123") == "sk-test-123"


def test_resolve_api_key_from_file(tmp_path: Path) -> None:
    p = tmp_path / "key.txt"
    p.write_text("sk-file-456\n", encoding="utf-8")
    assert _resolve_api_key(p) == "sk-file-456"
