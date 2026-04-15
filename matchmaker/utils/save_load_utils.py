from pathlib import Path
from collections.abc import Sequence

from pydantic import BaseModel

from matchmaker.data_models import CVRecord, JobRecord


def save_records(records: Sequence[BaseModel], path: str | Path) -> None:
    """Append-write a list of Pydantic records to a JSONL file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def load_job_records(path: str | Path) -> list[JobRecord]:
    return [
        JobRecord.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def load_cv_records(path: str | Path) -> list[CVRecord]:
    return [
        CVRecord.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
