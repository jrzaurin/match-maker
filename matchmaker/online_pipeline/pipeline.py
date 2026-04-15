"""Online matching pipeline — PDF CV → top-N ranked job matches."""

import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from matchmaker.config import TOP_N, RESULTS_DIR, PROCESSED_DATA_DIR
from matchmaker.data_models import CVRecord, JobRecord, RankedMatchResult
from matchmaker.extractors.extract import ExtractionPipeline
from matchmaker.utils.prompt_builders import build_match_user
from matchmaker.utils.save_load_utils import load_job_records
from matchmaker.prompts.cv_to_job_match import match_ranker_system
from matchmaker.extractors.base_extractor import Extractor
from matchmaker.online_pipeline.cv_reader import read_pdf
from matchmaker.extractors.ollama_extractor import OllamaExtractor
from matchmaker.extractors.openai_extractor import OpenAIExtractor
from matchmaker.offline_pipeline.vectorizer import TitleEncoder


class OnlinePipeline:
    """End-to-end online pipeline:

    1. Read CV from PDF.
    2. Extract structured ``CVRecord`` via LLM.
    3. First-pass retrieval: TF-IDF cosine similarity on job title → TOP_N candidates.
    4. LLM ranking: score and rank the TOP_N with a rationale.
    5. Save results to ``results/`` and optionally print.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n_workers: int = 3,
        jobs_path: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self._model = model
        self._extraction_pipeline = ExtractionPipeline(model=model, n_workers=n_workers)
        self._jobs_path = (
            Path(jobs_path) if jobs_path else Path(PROCESSED_DATA_DIR) / "jobs.jsonl"
        )
        self._verbose = verbose

    async def match(
        self,
        cv_path: str | Path,
        print_results: bool = True,
    ) -> RankedMatchResult:
        cv_path = Path(cv_path)

        # Step 1 — read PDF
        if self._verbose:
            print(f"[1/4] Reading CV from {cv_path.name} ...")

        body = read_pdf(cv_path)

        # Step 2 — extract CV info
        if self._verbose:
            print("[2/4] Extracting CV info via LLM ...")

        cv_record = await self._extraction_pipeline.extract_cv(
            filename=cv_path.name, body=body
        )

        if self._verbose:
            print(f"      → {cv_record.name} | {cv_record.most_recent_job_title}")

        # Step 3 — first-pass title retrieval
        if self._verbose:
            print("[3/4] Running first-pass title retrieval ...")

        top_jobs = self._retrieve_top_n(cv_record)

        if self._verbose:
            print(f"      → Top {len(top_jobs)} candidates:")
            for j in top_jobs:
                print(f"        {j.id}: {j.title}")

        # Step 4 — LLM ranking
        if self._verbose:
            print("[4/4] Ranking with LLM ...")

        result = await self._rank(cv_record, top_jobs)

        # Save + optionally display
        self._save(cv_path.stem, result)
        if print_results:
            self._print(result)

        return result

    def _retrieve_top_n(self, cv: CVRecord) -> list[JobRecord]:
        jobs = load_job_records(self._jobs_path)
        encoder = TitleEncoder.load()

        job_matrix = encoder.encode_many([j.title for j in jobs])
        cv_vec = encoder.encode_one(cv.most_recent_job_title)

        scores = cosine_similarity(cv_vec, job_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:TOP_N]
        return [jobs[int(i)] for i in top_idx]

    async def _rank(self, cv: CVRecord, jobs: list[JobRecord]) -> RankedMatchResult:
        _OLLAMA_PREFIXES = ("qwen", "deepseek", "llama", "phi", "gemma", "mistral")
        extractor: Extractor = (
            OllamaExtractor(model=self._model)
            if any(self._model.startswith(p) for p in _OLLAMA_PREFIXES)
            else OpenAIExtractor(model=self._model)
        )

        # The prompt template contains both instructions and data placeholders,
        # so format the full prompt as the user turn with a minimal system turn.
        system = match_ranker_system
        user = self._build_ranking_user(cv, jobs)
        return await extractor.extract_one(
            system=system,
            user=user,
            model_cls=RankedMatchResult,
        )

    @staticmethod
    def _build_ranking_user(cv: CVRecord, jobs: list[JobRecord]) -> str:
        cv_json = json.dumps(cv.model_dump(mode="json"), indent=2)
        jobs_json = json.dumps([j.model_dump(mode="json") for j in jobs], indent=2)
        return build_match_user(cv_json, jobs_json)

    def _save(self, stem: str, result: RankedMatchResult) -> None:
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_path = Path(RESULTS_DIR) / f"{stem}_{ts}.json"
        out_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        print(f"\nResults saved → {out_path}")

    @staticmethod
    def _print(result: RankedMatchResult) -> None:
        print("\n" + "=" * 60)
        print("MATCH RESULTS")
        print("=" * 60)
        for entry in result.ranked_jobs:
            print(f"\n#{entry.rank}  {entry.job_id}  (score: {entry.score}/100)")
            print(f"   {entry.rationale}")
        print("\n--- Summary ---")
        print(f"Best fit : {result.summary.best_fit_job_id}")
        print(f"Comment  : {result.summary.overall_comment}")
        print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m matchmaker.online_pipeline.matcher <path/to/cv.pdf>")
        sys.exit(1)

    asyncio.run(OnlinePipeline().match(sys.argv[1]))
