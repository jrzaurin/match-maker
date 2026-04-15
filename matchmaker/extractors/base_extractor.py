import asyncio
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from matchmaker.utils.jsonutil import parse_json_loose

T = TypeVar("T", bound=BaseModel)


class Extractor(ABC):
    """Abstract async extractor with bounded concurrency and validation retry.

    Subclasses implement `_complete_raw` for the actual model transport.
    """

    def __init__(self, n_workers: int = 5, max_attempts: int = 3) -> None:
        self._semaphore = asyncio.Semaphore(n_workers)
        self._max_attempts = max_attempts

    @abstractmethod
    async def _complete_raw(self, *, system: str, user: str) -> str:
        """Single async call to the model; returns raw text."""

    async def extract_one(
        self,
        *,
        system: str,
        user: str,
        model_cls: type[T],
    ) -> T:
        async with self._semaphore:
            return await self._call_with_retry(
                system=system, user=user, model_cls=model_cls
            )

    async def extract_many(
        self,
        items: list[dict],  # each: {"system": str, "user": str, "model_cls": type[T]}
    ) -> list[BaseModel]:
        tasks = [
            self.extract_one(
                system=item["system"],
                user=item["user"],
                model_cls=item["model_cls"],
            )
            for item in items
        ]
        return list[BaseModel](await asyncio.gather(*tasks))

    async def _call_with_retry(
        self,
        *,
        system: str,
        user: str,
        model_cls: type[T],
    ) -> T:
        suffix = ""
        last_err: Exception | None = None
        for _ in range(self._max_attempts):
            raw = await self._complete_raw(system=system, user=user + suffix)
            try:
                data = parse_json_loose(raw)
                if not isinstance(data, dict):
                    raise TypeError("Expected JSON object")
                return model_cls.model_validate(data)
            except (ValidationError, TypeError, ValueError) as e:
                last_err = e
                suffix = (
                    "\n\nYour previous output was invalid. "
                    "Reply with ONE corrected JSON object only.\n"
                    f"Error: {e}\n"
                )
        assert last_err is not None
        raise last_err
