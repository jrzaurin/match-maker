# NOTE: many comments can be made here. There are a number of ways to speed
# inference that can be discussed in detail in due time. One example is using
# vLLM to speed up the inference. Another is, of course, using a different
# method where we encode consistenty CVs and job descriptions in a way that the
# retrieval can happen based on distance over encoded vectors rather than "brute
# forcing" and LLM
import json
import asyncio
from enum import Enum
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from matchmaker.vocab import RemotePreference
from matchmaker.config import TOP_N, RESULTS_DIR, PROCESSED_DATA_DIR
from matchmaker.data_models import (
    CVRecord,
    JobRecord,
    RankedJob,
    RankedMatchResult,
    CandidatePreferences,
)
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
        api_key: str | Path | None = None,
        n_workers: int = 3,
        jobs_path: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self._model = model
        self._extraction_pipeline = ExtractionPipeline(
            model=model, n_workers=n_workers, api_key=api_key
        )
        self._jobs_path = (
            Path(jobs_path) if jobs_path else Path(PROCESSED_DATA_DIR) / "jobs.jsonl"
        )
        self._verbose = verbose

    async def match(
        self,
        cv_path: str | Path,
        preferences: CandidatePreferences | None = None,
        print_results: bool = True,
    ) -> RankedMatchResult:
        cv_path = Path(cv_path)
        n_steps = "5" if preferences else "4"

        # Step 1 — read PDF
        if self._verbose:
            print(f"[1/{n_steps}] Reading CV from {cv_path.name} ...")
        body = read_pdf(cv_path)

        # Step 2 — extract CV info
        if self._verbose:
            print(f"[2/{n_steps}] Extracting CV info via LLM ...")
        cv_record = await self._extraction_pipeline.extract_cv(
            filename=cv_path.name, body=body
        )
        if self._verbose:
            print(f"      → {cv_record.name} | {cv_record.most_recent_job_title}")

        # Step 3 — first-pass title retrieval
        if self._verbose:
            print(f"[3/{n_steps}] Running first-pass title retrieval ...")
        top_jobs = self._retrieve_top_n(cv_record)
        if self._verbose:
            print(f"      → Top {len(top_jobs)} candidates:")
            for j in top_jobs:
                print(f"        {j.id}: {j.title}")

        # Step 4 — LLM ranking
        if self._verbose:
            print(f"[4/{n_steps}] Ranking with LLM ...")
        result = await self._rank(cv_record, top_jobs)

        # Step 5 — preference re-ranking (optional)
        if preferences is not None:
            if self._verbose:
                print(f"[5/{n_steps}] Applying candidate preferences ...")
            jobs_by_id = {j.id: j for j in top_jobs}
            result = self._apply_preferences(result, jobs_by_id, preferences)

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
        _OLLAMA_PREFIXES = ("qwen", "deepseek")
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
    def _apply_preferences(
        result: RankedMatchResult,
        jobs_by_id: dict[str, JobRecord],
        prefs: CandidatePreferences,
    ) -> RankedMatchResult:
        """Re-rank the LLM result using hard/soft preference filters.

        Hard preferences (role type, remote policy, salary) carry more weight.
        Soft preferences (seniority, company stage/size, location) carry less.
        Jobs with fewer unmet preferences float to the top; LLM score breaks ties.
        """

        def misses(entry: RankedJob) -> tuple[int, int, float]:
            job = jobs_by_id.get(entry.job_id)
            if job is None:
                return (99, 99, -entry.score)
            hard, soft = 0, 0

            if (
                prefs.preferred_role_types
                and job.role_type not in prefs.preferred_role_types
            ):
                hard += 1

            if (
                prefs.remote_preference != RemotePreference.no_preference
                and job.remote_policy != prefs.remote_preference
            ):
                hard += 1

            if (
                prefs.min_salary is not None
                and job.salary_min is not None
                and job.salary_min < prefs.min_salary
            ):
                hard += 1

            if (
                prefs.preferred_seniority
                and job.seniority_level not in prefs.preferred_seniority
            ):
                soft += 1

            if (
                prefs.preferred_company_stages
                and job.company_stage not in prefs.preferred_company_stages
            ):
                soft += 1

            if (
                prefs.preferred_company_sizes
                and job.company_size not in prefs.preferred_company_sizes
            ):
                soft += 1

            if prefs.preferred_locations:
                loc_hit = any(
                    loc.lower() in job.location.lower()
                    for loc in prefs.preferred_locations
                )
                if not loc_hit:
                    soft += 1

            return (hard, soft, -entry.score)  # ascending sort → fewer misses first

        reranked = sorted(result.ranked_jobs, key=misses)
        reranked = [
            entry.model_copy(update={"rank": i + 1})
            for i, entry in enumerate[RankedJob](reranked)
        ]
        best_id = reranked[0].job_id
        return result.model_copy(
            update={
                "ranked_jobs": reranked,
                "summary": result.summary.model_copy(
                    update={"best_fit_job_id": best_id}
                ),
            }
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


def _parse_enum_list(raw: list[str], enum_cls: type[Enum]) -> list:
    """Convert a list of raw strings to enum members, raising on unknown values."""
    out = []
    for v in raw:
        try:
            out.append(enum_cls(v))
        except ValueError:
            valid = ", ".join(str(e.value) for e in enum_cls)
            raise SystemExit(
                f"Invalid value '{v}' for {enum_cls.__name__}. Choose from: {valid}"
            )
    return out


def cli() -> None:
    import typer

    from matchmaker.vocab import (
        RoleType,
        CompanySize,
        CompanyStage,
        SeniorityLevel,
        RemotePreference,
    )

    app = typer.Typer(help="Match a CV PDF against the job corpus.")

    @app.command()
    def match(
        api_key: Optional[str] = typer.Option(None, help="API key for the LLM model."),
        cv_path: str = typer.Argument(..., help="Path to the CV PDF file."),
        model: str = typer.Option(
            "gpt-4o-mini", help="LLM model name (OpenAI or Ollama)."
        ),
        n_workers: int = typer.Option(3, help="Number of concurrent LLM workers."),
        role_types: list[str] = typer.Option(
            [],
            "--role-type",
            help=f"Preferred role types. Allowed: {', '.join(e.value for e in RoleType)}",
        ),
        remote: str = typer.Option(
            "no_preference",
            help=f"Remote preference. Allowed: {', '.join(e.value for e in RemotePreference)}",
        ),
        min_salary: int | None = typer.Option(
            None, help="Minimum acceptable salary (GBP)."
        ),
        seniority: list[str] = typer.Option(
            [],
            "--seniority",
            help=f"Preferred seniority levels. Allowed: {', '.join(e.value for e in SeniorityLevel)}",
        ),
        company_stages: list[str] = typer.Option(
            [],
            "--company-stage",
            help=f"Preferred company stages. Allowed: {', '.join(e.value for e in CompanyStage)}",
        ),
        company_sizes: list[str] = typer.Option(
            [],
            "--company-size",
            help=f"Preferred company sizes. Allowed: {', '.join(e.value for e in CompanySize)}",
        ),
        locations: list[str] = typer.Option(
            [], "--location", help="Preferred locations (partial match, e.g. 'London')."
        ),
        no_print: bool = typer.Option(
            False, "--no-print", help="Suppress terminal output of results."
        ),
    ) -> None:
        prefs = None
        has_prefs = any(
            [
                role_types,
                remote != "no_preference",
                min_salary,
                seniority,
                company_stages,
                company_sizes,
                locations,
            ]
        )

        if has_prefs:
            prefs = CandidatePreferences(
                preferred_role_types=_parse_enum_list(role_types, RoleType),
                remote_preference=RemotePreference(remote),
                min_salary=min_salary,
                preferred_seniority=_parse_enum_list(seniority, SeniorityLevel),
                preferred_company_stages=_parse_enum_list(company_stages, CompanyStage),
                preferred_company_sizes=_parse_enum_list(company_sizes, CompanySize),
                preferred_locations=locations,
            )

        pipeline = OnlinePipeline(model=model, n_workers=n_workers, api_key=api_key)
        asyncio.run(
            pipeline.match(cv_path, preferences=prefs, print_results=not no_print)
        )

    app()


if __name__ == "__main__":
    # # No preferences — just LLM ranking
    # python -m matchmaker.online_pipeline.pipeline cv.pdf

    # # With preferences
    # python -m matchmaker.online_pipeline.pipeline cv.pdf \
    # --role-type ml_engineer \
    # --remote hybrid \
    # --min-salary 100000 \
    # --seniority senior --seniority lead \
    # --company-stage scale_up \
    # --location London

    # # Suppress terminal output (just save to results/)
    # python -m matchmaker.online_pipeline.pipeline cv.pdf --no-print

    # # Use a different model
    # python -m matchmaker.online_pipeline.pipeline cv.pdf --model qwen3.5:9b

    cli()
