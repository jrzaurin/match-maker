import os

from openai import AsyncOpenAI

from matchmaker.tokens_and_api_keys import open_ai_api_key
from matchmaker.extractors.base_extractor import Extractor


class OpenAIExtractor(Extractor):
    """Extractor backed by OpenAI (gpt-4o-mini by default)."""

    def __init__(
        self,
        model: str | None = None,
        n_workers: int = 5,
        max_attempts: int = 3,
    ) -> None:
        super().__init__(n_workers=n_workers, max_attempts=max_attempts)
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = AsyncOpenAI(api_key=open_ai_api_key)

    async def _complete_raw(self, *, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned empty content")
        return content


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from matchmaker.config import CV_DIR, JOB_DIR, RAW_DATA_DIR
    from matchmaker.data_models import CVRecord, JobRecord
    from matchmaker.utils.prompt_builders import (
        user_extract_cv,
        user_extract_job,
        build_extract_cv_system,
        build_extract_job_system,
    )

    job_filename = "zoopla_data_analyst.txt"
    cv_filename = "cv1.txt"
    job_spec_path = Path(RAW_DATA_DIR) / JOB_DIR / job_filename
    cv_spec_path = Path(RAW_DATA_DIR) / CV_DIR / cv_filename
    job_body = job_spec_path.read_text()
    cv_body = cv_spec_path.read_text()

    openai_extractor = OpenAIExtractor()
    openai_job_record = asyncio.run(
        openai_extractor.extract_one(
            system=build_extract_job_system(),
            user=user_extract_job(job_filename, job_body),
            model_cls=JobRecord,
        )
    )
    print(openai_job_record.model_dump(mode="json"))

    openai_cv_record = asyncio.run(
        openai_extractor.extract_one(
            system=build_extract_cv_system(),
            user=user_extract_cv(cv_filename, cv_body),
            model_cls=CVRecord,
        )
    )
    print(openai_cv_record.model_dump(mode="json"))
