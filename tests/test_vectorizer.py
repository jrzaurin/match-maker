from pathlib import Path

import pytest
from sklearn.metrics.pairwise import cosine_similarity

from matchmaker.data_models import JobRecord
from matchmaker.utils.save_load_utils import save_records
from matchmaker.offline_pipeline.vectorizer import TitleEncoder, _normalize


def test_normalize_lowercases_and_strips_non_alnum() -> None:
    assert _normalize("Senior ML Engineer (Hands-On)!") == "senior ml engineer handson"


def test_fit_save_load_and_encode_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate artifacts/ and processed_data/ to this test dir
    monkeypatch.chdir(tmp_path)

    processed = tmp_path / "processed_data"
    processed.mkdir()
    jobs_path = processed / "jobs.jsonl"

    jobs = [
        JobRecord(
            id="j1",
            title="Senior Machine Learning Engineer",
            company="A",
            industry="adtech",
            location="London",
            remote_policy="hybrid",
            seniority_level="senior",
            role_type="ml_engineer",
            company_stage="scale_up",
            company_size="medium",
        ),
        JobRecord(
            id="j2",
            title="Data Analyst",
            company="B",
            industry="retail",
            location="London",
            remote_policy="hybrid",
            seniority_level="mid",
            role_type="data_analyst",
            company_stage="enterprise",
            company_size="large",
        ),
    ]
    save_records(jobs, jobs_path)

    enc = TitleEncoder()
    enc.fit_and_save(jobs_path)

    # Ensure pickle created
    assert (tmp_path / "artifacts" / "title_vectorizer.pkl").exists()

    enc2 = TitleEncoder.load()
    job_mat = enc2.encode_many([j.title for j in jobs])
    cv_vec = enc2.encode_one("Senior ML Engineer")

    assert job_mat.shape[0] == 2
    assert cv_vec.shape[0] == 1
    assert job_mat.shape[1] == cv_vec.shape[1]


def test_similar_titles_score_higher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    processed = tmp_path / "processed_data"
    processed.mkdir()
    jobs_path = processed / "jobs.jsonl"

    jobs = [
        JobRecord(
            id="j1",
            title="Senior Machine Learning Engineer",
            company="A",
            industry="adtech",
            location="London",
            remote_policy="hybrid",
            seniority_level="senior",
            role_type="ml_engineer",
            company_stage="scale_up",
            company_size="medium",
        ),
        JobRecord(
            id="j2",
            title="Retail Store Manager",
            company="B",
            industry="retail",
            location="London",
            remote_policy="on_site",
            seniority_level="mid",
            role_type="ml_manager",
            company_stage="enterprise",
            company_size="large",
        ),
    ]
    save_records(jobs, jobs_path)

    enc = TitleEncoder()
    enc.fit_and_save(jobs_path)
    enc = TitleEncoder.load()

    job_mat = enc.encode_many([j.title for j in jobs])
    q = enc.encode_one("Machine Learning Engineer")
    scores = cosine_similarity(q, job_mat).flatten()

    assert scores[0] > scores[1]
