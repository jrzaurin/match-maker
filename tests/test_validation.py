from matchmaker.data_models import CVRecord, JobRecord


def test_job_domain_expertise_normalizes_hyphens_spaces_and_synonyms() -> None:
    j = JobRecord(
        id="j1",
        title="t",
        company="c",
        industry="i",
        location="loc",
        remote_policy="hybrid",
        seniority_level="senior",
        role_type="ml_engineer",
        company_stage="scale_up",
        company_size="medium",
        domain_expertise_required=[
            "Credit Risk",  # spaces -> underscore
            "fraud-detection",  # hyphen -> underscore
            "ads",  # synonym -> adtech
        ],
    )
    assert j.domain_expertise_required == ["credit_risk", "fraud_detection", "adtech"]


def test_cv_domain_expertise_unknown_falls_back_to_general_and_deduplicates() -> None:
    cv = CVRecord(
        id="cv1",
        name="n",
        location="loc",
        total_years_experience=1,
        most_recent_job_title="t",
        seniority_level="mid",
        primary_role_type="data_scientist",
        domain_expertise=[
            "fintech",
            "FinTech",  # normalizes to fintech and should dedupe
            "some_unknown_domain",
            "some_unknown_domain",  # dedupe
        ],
    )
    assert cv.domain_expertise == ["fintech", "general_ml_ai"]


def test_domain_expertise_none_or_empty_becomes_empty_list() -> None:
    j = JobRecord(
        id="j1",
        title="t",
        company="c",
        industry="i",
        location="loc",
        remote_policy="hybrid",
        seniority_level="senior",
        role_type="ml_engineer",
        company_stage="scale_up",
        company_size="medium",
        domain_expertise_required=None,  # type: ignore[arg-type]
    )
    assert j.domain_expertise_required == []
