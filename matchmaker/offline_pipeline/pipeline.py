"""Offline pipeline — extract job specs from disk and build the title index."""

import asyncio
from pathlib import Path

import typer

from matchmaker.config import JOB_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from matchmaker.data_models import JobRecord
from matchmaker.extractors.extract import ExtractionPipeline
from matchmaker.utils.save_load_utils import save_records
from matchmaker.offline_pipeline.vectorizer import TitleEncoder


class OfflinePipeline:
    """Two-step offline pipeline:

    1. ``run_extraction`` — read raw .txt job specs, extract structured
       ``JobRecord``s via LLM, and persist to ``processed_data/jobs.jsonl``.

    2. ``run_indexing`` — load the persisted records, fit a TF-IDF
       ``TitleEncoder`` on job titles, and save it to ``artifacts/``.

    Run both in sequence with ``run_all``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | Path | None = None,
        n_workers: int = 3,
        jobs_jsonl: str | Path | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._n_workers = n_workers
        self._jobs_path = (
            Path(jobs_jsonl) if jobs_jsonl else Path(PROCESSED_DATA_DIR) / "jobs.jsonl"
        )

    async def run_all(self) -> None:
        await self.run_extraction()
        self.run_indexing()

    async def run_extraction(self) -> list[JobRecord]:
        """Read .txt files, extract via LLM, save JSONL. Returns the records."""
        job_dir = Path(RAW_DATA_DIR) / JOB_DIR
        files = [
            (p.name, p.read_text(encoding="utf-8"))
            for p in sorted(job_dir.glob("*.txt"))
        ]

        if not files:
            raise FileNotFoundError(f"No .txt files found in {job_dir}")

        print(
            f"[step 1/2] Extracting {len(files)} job specs  model={self._model}  workers={self._n_workers}"
        )

        pipeline = ExtractionPipeline(
            model=self._model, n_workers=self._n_workers, api_key=self._api_key
        )
        jobs = await pipeline.extract_jobs(files)

        self._jobs_path.parent.mkdir(parents=True, exist_ok=True)
        save_records(jobs, self._jobs_path)
        print(f"          Saved {len(jobs)} records → {self._jobs_path}")

        return jobs

    def run_indexing(self) -> TitleEncoder:
        """Fit TF-IDF on job titles from the JSONL and save the encoder."""
        print(f"[step 2/2] Fitting title encoder from {self._jobs_path}")

        encoder = TitleEncoder()
        encoder.fit_and_save(self._jobs_path)
        print("          Title encoder saved.")

        return encoder


def run_offline_pipeline(
    model: str = typer.Option("gpt-4o-mini", help="LLM model name."),
    api_key: str | None = typer.Option(
        None, help="OpenAI API key or path to a key file."
    ),
    n_workers: int = typer.Option(3, help="Number of concurrent LLM workers."),
    index_only: bool = typer.Option(
        False, "--index-only", help="Skip extraction, only rebuild the title index."
    ),
) -> None:
    """Extract job specs from disk and build the TF-IDF title index."""
    pipeline = OfflinePipeline(model=model, api_key=api_key, n_workers=n_workers)
    if index_only:
        pipeline.run_indexing()
    else:
        asyncio.run(pipeline.run_all())


if __name__ == "__main__":

    # example usage:
    # python -m matchmaker.offline_pipeline.pipeline
    # --model qwen3.5:9b
    # --api-key ~/.openai/api_key
    # --n-workers 3
    # --index-only

    typer.run(run_offline_pipeline)
