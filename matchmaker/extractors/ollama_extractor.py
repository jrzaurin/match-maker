import asyncio

from ollama import AsyncClient as OllamaClient

from matchmaker.extractors.base_extractor import Extractor


class OllamaExtractor(Extractor):
    """Extractor backed by a local Ollama model (qwen3:8b by default)."""

    def __init__(
        self,
        model: str = "qwen3.5:2b",
        n_workers: int = 3,
        max_attempts: int = 3,
    ) -> None:
        super().__init__(n_workers=n_workers, max_attempts=max_attempts)
        self._model = model

    async def _complete_raw(self, *, system: str, user: str) -> str:
        client = OllamaClient()
        resp = await client.chat(
            model=self._model,
            format="json",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.message.content
        if not content:
            raise RuntimeError("Ollama returned empty content")
        return content


if __name__ == "__main__":
    from pathlib import Path

    from matchmaker.config import CV_DIR, JOB_DIR, RAW_DATA_DIR
    from matchmaker.data_models import CVRecord, JobRecord
    from matchmaker.utils.prompt_builders import (
        user_extract_cv,
        user_extract_job,
        build_extract_cv_system,
        build_extract_job_system,
    )

    job_filename = "casra_data_scientist.txt"
    cv_filename = "cv1.txt"
    job_spec_path = Path(RAW_DATA_DIR) / JOB_DIR / job_filename
    cv_spec_path = Path(RAW_DATA_DIR) / CV_DIR / cv_filename
    job_body = job_spec_path.read_text()
    cv_body = cv_spec_path.read_text()

    ollama_extractor = OllamaExtractor()
    ollama_job_record = asyncio.run(
        ollama_extractor.extract_one(
            system=build_extract_job_system(),
            user=user_extract_job(job_filename, job_body),
            model_cls=JobRecord,
        )
    )
    print(ollama_job_record.model_dump(mode="json"))

    ollama_cv_record = asyncio.run(
        ollama_extractor.extract_one(
            system=build_extract_cv_system(),
            user=user_extract_cv(cv_filename, cv_body),
            model_cls=CVRecord,
        )
    )
    print(ollama_cv_record.model_dump(mode="json"))
