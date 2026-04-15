"""Build system prompts from templates + vocab appendix + few-shot."""

from matchmaker.vocab import enum_appendix
from matchmaker.prompts import parameterize_cv, parameterize_job


def build_extract_job_system() -> str:
    return parameterize_job.replace("{{ENUM_APPENDIX}}", enum_appendix())


def build_extract_cv_system() -> str:
    return parameterize_cv.replace("{{ENUM_APPENDIX}}", enum_appendix())


def user_extract_job(filename: str, body: str) -> str:
    return (
        f"Source filename: {filename}\n\n"
        "Job description text:\n---\n"
        f"{body.strip()}\n---\n"
        "Return one JSON object matching the schema."
    )


def user_extract_cv(filename: str, body: str) -> str:
    return (
        f"Source filename: {filename}\n\n"
        "CV text:\n---\n"
        f"{body.strip()}\n---\n"
        "Return one JSON object matching the schema."
    )


def build_match_user(cv_summary: str, job_descriptions: str) -> str:
    return (
        "Candidate CV (JSON):\n"
        f"{cv_summary}\n\n"
        "Job Descriptions (JSON array):\n"
        f"{job_descriptions}"
    )
