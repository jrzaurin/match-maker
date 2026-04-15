from matchmaker.vocab import RemotePreference
from matchmaker.data_models import (
    JobRecord,
    RankedJob,
    MatchSummary,
    RankedMatchResult,
    CandidatePreferences,
)
from matchmaker.online_pipeline.pipeline import OnlinePipeline


def _job(
    *,
    id: str,
    role_type: str,
    remote_policy: str,
    salary_min: int | None,
    location: str = "London",
) -> JobRecord:
    return JobRecord(
        id=id,
        title=id,
        company="c",
        industry="i",
        location=location,
        remote_policy=remote_policy,
        salary_min=salary_min,
        seniority_level="senior",
        role_type=role_type,
        company_stage="scale_up",
        company_size="medium",
    )


def test_apply_preferences_hard_constraints_dominate_score_tie_break() -> None:
    jobs = [
        _job(
            id="a", role_type="ml_engineer", remote_policy="hybrid", salary_min=120000
        ),
        _job(
            id="b", role_type="data_analyst", remote_policy="hybrid", salary_min=120000
        ),
    ]
    jobs_by_id = {j.id: j for j in jobs}

    # LLM liked job b more initially
    result = RankedMatchResult(
        ranked_jobs=[
            RankedJob(job_id="b", score=90, rank=1, rationale=""),
            RankedJob(job_id="a", score=80, rank=2, rationale=""),
        ],
        summary=MatchSummary(best_fit_job_id="b", overall_comment=""),
    )

    prefs = CandidatePreferences(preferred_role_types=["ml_engineer"])
    out = OnlinePipeline._apply_preferences(result, jobs_by_id, prefs)

    assert out.ranked_jobs[0].job_id == "a"
    assert [r.rank for r in out.ranked_jobs] == [1, 2]
    assert out.summary.best_fit_job_id == "a"


def test_apply_preferences_salary_and_remote_preference() -> None:
    jobs = [
        _job(id="a", role_type="ml_engineer", remote_policy="hybrid", salary_min=90000),
        _job(
            id="b",
            role_type="ml_engineer",
            remote_policy="fully_remote",
            salary_min=130000,
        ),
    ]
    jobs_by_id = {j.id: j for j in jobs}
    result = RankedMatchResult(
        ranked_jobs=[
            RankedJob(job_id="a", score=95, rank=1, rationale=""),
            RankedJob(job_id="b", score=80, rank=2, rationale=""),
        ],
        summary=MatchSummary(best_fit_job_id="a", overall_comment=""),
    )

    prefs = CandidatePreferences(
        remote_preference=RemotePreference.fully_remote, min_salary=100000
    )
    out = OnlinePipeline._apply_preferences(result, jobs_by_id, prefs)
    assert out.ranked_jobs[0].job_id == "b"
