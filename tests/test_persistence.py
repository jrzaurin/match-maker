from pathlib import Path

from matchmaker.data_models import CVRecord, JobRecord
from matchmaker.utils.save_load_utils import (
    save_records,
    load_cv_records,
    load_job_records,
)


def test_save_and_load_job_records_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "jobs.jsonl"
    jobs = [
        JobRecord(
            id="j1",
            title="Senior Data Scientist",
            company="Acme",
            industry="fintech",
            location="London",
            remote_policy="hybrid",
            seniority_level="senior",
            role_type="data_scientist",
            company_stage="scale_up",
            company_size="medium",
            domain_expertise_required=["credit_risk"],
        ),
        JobRecord(
            id="j2",
            title="ML Engineer",
            company="Beta",
            industry="adtech",
            location="Remote",
            remote_policy="fully_remote",
            seniority_level="mid",
            role_type="ml_engineer",
            company_stage="startup",
            company_size="small",
        ),
    ]

    save_records(jobs, out)
    loaded = load_job_records(out)

    assert [j.model_dump(mode="json") for j in loaded] == [
        j.model_dump(mode="json") for j in jobs
    ]


def test_save_and_load_cv_records_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "cvs.jsonl"
    cvs = [
        CVRecord(
            id="cv1",
            name="Jane Doe",
            location="UK",
            total_years_experience=5,
            most_recent_job_title="Data Scientist",
            seniority_level="mid",
            primary_role_type="data_scientist",
            domain_expertise=["fintech"],
        )
    ]

    save_records(cvs, out)
    loaded = load_cv_records(out)

    assert loaded[0].model_dump(mode="json") == cvs[0].model_dump(mode="json")


def test_load_ignores_blank_lines(tmp_path: Path) -> None:
    out = tmp_path / "jobs.jsonl"
    out.write_text(
        '{"id":"j1","title":"t","company":"c","industry":"i","location":"l","remote_policy":"hybrid","seniority_level":"mid","role_type":"ml_engineer","company_stage":"startup","company_size":"small","domain_expertise_required":[],"key_responsibilities":[],"required_skills":[],"nice_to_have_skills":[],"leadership_required":{"required":false,"team_size_hint":null},"visa_sponsorship":false,"days_in_office_per_week":null,"salary_min":null,"salary_max":null,"source_file":null,"extracted_at":null}\n\n',
        encoding="utf-8",
    )
    loaded = load_job_records(out)
    assert len(loaded) == 1
