# NOTE: the code here and in general could be a bit better organized and less
# verbose. In terms of performance, here in particular, a local run via
# huggingface might be better.
import asyncio
from pathlib import Path

from matchmaker.data_models import CVRecord, JobRecord, utc_now_iso
from matchmaker.utils.prompt_builders import (
    user_extract_cv,
    user_extract_job,
    build_extract_cv_system,
    build_extract_job_system,
)
from matchmaker.extractors.base_extractor import Extractor
from matchmaker.extractors.ollama_extractor import OllamaExtractor
from matchmaker.extractors.openai_extractor import OpenAIExtractor

_OLLAMA_PREFIXES = ("qwen", "deepseek")


def _resolve_api_key(api_key: str | Path | None) -> str | None:
    """Resolve an API key from a raw string, a file path, or None (use default)."""
    if api_key is None:
        return None
    p = Path(api_key)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return str(api_key)


def _make_extractor(
    model: str,
    n_workers: int,
    max_attempts: int,
    api_key: str | None = None,
) -> Extractor:
    if any(model.startswith(p) for p in _OLLAMA_PREFIXES):
        return OllamaExtractor(
            model=model, n_workers=n_workers, max_attempts=max_attempts
        )
    return OpenAIExtractor(
        model=model, n_workers=n_workers, max_attempts=max_attempts, api_key=api_key
    )


class ExtractionPipeline:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n_workers: int = 5,
        max_attempts: int = 3,
        api_key: str | Path | None = None,
    ) -> None:
        self._extractor = _make_extractor(
            model, n_workers, max_attempts, _resolve_api_key(api_key)
        )

    async def extract_job(self, *, filename: str, body: str) -> JobRecord:
        rec = await self._extractor.extract_one(
            system=build_extract_job_system(),
            user=user_extract_job(filename, body),
            model_cls=JobRecord,
        )
        return rec.model_copy(
            update={
                "id": Path(filename).stem,
                "source_file": filename,
                "extracted_at": utc_now_iso(),
            }
        )

    async def extract_cv(self, *, filename: str, body: str) -> CVRecord:
        rec = await self._extractor.extract_one(
            system=build_extract_cv_system(),
            user=user_extract_cv(filename, body),
            model_cls=CVRecord,
        )
        return rec.model_copy(
            update={
                "id": Path(filename).stem,
                "source_file": filename,
                "extracted_at": utc_now_iso(),
            }
        )

    async def extract_jobs(self, files: list[tuple[str, str]]) -> list[JobRecord]:
        """files: list of (filename, body) pairs."""
        return list[JobRecord](
            await asyncio.gather(
                *[self.extract_job(filename=fn, body=body) for fn, body in files]
            )
        )

    async def extract_cvs(self, files: list[tuple[str, str]]) -> list[CVRecord]:
        return list[CVRecord](
            await asyncio.gather(
                *[self.extract_cv(filename=fn, body=body) for fn, body in files]
            )
        )
