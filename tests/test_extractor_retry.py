import asyncio

import pytest
from pydantic import BaseModel

from matchmaker.extractors.base_extractor import Extractor


class _Out(BaseModel):
    a: int


class _FakeExtractor(Extractor):
    def __init__(self, responses: list[str], max_attempts: int = 3) -> None:
        super().__init__(n_workers=1, max_attempts=max_attempts)
        self._responses = responses
        self.calls: int = 0

    async def _complete_raw(self, *, system: str, user: str) -> str:
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


def test_retry_invalid_json_then_success() -> None:
    ex = _FakeExtractor(["{bad json", '{"a": 1}'], max_attempts=3)
    out = asyncio.run(ex.extract_one(system="s", user="u", model_cls=_Out))
    assert out.a == 1
    assert ex.calls == 2


def test_retry_json_not_object_then_success() -> None:
    ex = _FakeExtractor(["[1,2,3]", '{"a": 2}'], max_attempts=3)
    out = asyncio.run(ex.extract_one(system="s", user="u", model_cls=_Out))
    assert out.a == 2
    assert ex.calls == 2


def test_exhaust_attempts_raises_last_error() -> None:
    ex = _FakeExtractor(["{bad", "{still bad", "{nope"], max_attempts=3)
    with pytest.raises(Exception):
        asyncio.run(ex.extract_one(system="s", user="u", model_cls=_Out))


def test_extract_many_preserves_order() -> None:
    ex = _FakeExtractor(['{"a": 1}', '{"a": 2}', '{"a": 3}'], max_attempts=1)
    items = [
        {"system": "s", "user": "u1", "model_cls": _Out},
        {"system": "s", "user": "u2", "model_cls": _Out},
        {"system": "s", "user": "u3", "model_cls": _Out},
    ]
    outs = asyncio.run(ex.extract_many(items))
    assert [o.a for o in outs] == [1, 2, 3]
