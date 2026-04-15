import json

import pytest

from matchmaker.utils.jsonutil import parse_json_loose


def test_parse_json_loose_plain_json_object() -> None:
    assert parse_json_loose('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_json_loose_strips_whitespace() -> None:
    assert parse_json_loose('\n  {"a": 1}\n') == {"a": 1}


def test_parse_json_loose_fenced_json_with_language() -> None:
    text = """```json
{"a": 1}
```"""
    assert parse_json_loose(text) == {"a": 1}


def test_parse_json_loose_fenced_json_without_language() -> None:
    text = """```
{"a": 1}
```"""
    assert parse_json_loose(text) == {"a": 1}


def test_parse_json_loose_invalid_json_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_loose("{not json}")
