"""CLI: export schemas, ingest TXT → JSON corpus, match with preferences."""

import json
from pathlib import Path

import typer

from matchmaker.models import CVCorpusFile, JobCorpusFile, CandidatePreferences
from matchmaker.extract import (
    default_backend,
    extract_cv_from_text,
    extract_job_from_text,
)
from matchmaker.matcher import dump_match_json, match_cv_to_jobs
from matchmaker.schema_export import export_schemas

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("export-schemas")
def cmd_export_schemas(
    out: Path = typer.Option(
        Path("schemas"), "--out", "-o", help="Directory for JSON Schema files"
    ),
) -> None:
    """Write JSON Schema for job, CV, preferences, and match result."""
    export_schemas(out)
    typer.echo(f"Wrote JSON Schema files to {out.resolve()}")


@app.command("ingest-jobs")
def cmd_ingest_jobs(
    jobs_dir: Path = typer.Option(
        Path("data/job_descriptions"),
        "--jobs-dir",
        help="Directory containing job description .txt files",
    ),
    out: Path = typer.Option(
        Path("data/corpus/jobs.json"), "--out", help="Output JSON path"
    ),
    pattern: str = typer.Option(
        "*.txt", "--pattern", help="Glob pattern under jobs-dir"
    ),
) -> None:
    """Parameterize job TXT files into a JSON corpus (LLM extraction)."""
    backend = default_backend()
    paths = sorted(jobs_dir.glob(pattern))
    if not paths:
        typer.echo(f"No files matching {pattern!r} in {jobs_dir}", err=True)
        raise typer.Exit(code=1)
    jobs = []
    for p in paths:
        typer.echo(f"Extracting {p.name}...")
        body = p.read_text(encoding="utf-8")
        jobs.append(extract_job_from_text(backend, filename=p.name, body=body))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = JobCorpusFile(jobs=jobs)
    out.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote {len(jobs)} jobs to {out.resolve()}")


@app.command("ingest-cvs")
def cmd_ingest_cvs(
    cvs_dir: Path = typer.Option(
        Path("data/cvs"),
        "--cvs-dir",
        help="Directory containing CV .txt files",
    ),
    out: Path = typer.Option(Path("data/corpus/cvs.json"), "--out"),
    pattern: str = typer.Option("*.txt", "--pattern"),
) -> None:
    """Parameterize CV TXT files into a JSON corpus (LLM extraction)."""
    backend = default_backend()
    paths = sorted(cvs_dir.glob(pattern))
    if not paths:
        typer.echo(f"No files matching {pattern!r} in {cvs_dir}", err=True)
        raise typer.Exit(code=1)
    cvs = []
    for p in paths:
        typer.echo(f"Extracting {p.name}...")
        body = p.read_text(encoding="utf-8")
        cvs.append(extract_cv_from_text(backend, filename=p.name, body=body))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = CVCorpusFile(cvs=cvs)
    out.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote {len(cvs)} CVs to {out.resolve()}")


@app.command("match")
def cmd_match(
    jobs_path: Path = typer.Option(
        Path("data/corpus/jobs.json"), "--jobs", help="Parameterized jobs JSON"
    ),
    cvs_path: Path = typer.Option(
        Path("data/corpus/cvs.json"), "--cvs", help="Parameterized CVs JSON"
    ),
    cv_id: str = typer.Option(..., "--cv-id", help="CV id (filename stem, e.g. cv1)"),
    preferences_path: Path = typer.Option(
        Path("examples/prefs.json"),
        "--preferences",
        "-p",
        help="Candidate preferences JSON",
    ),
) -> None:
    """Rank all jobs for a CV using preferences (LLM)."""
    jobs_payload = JobCorpusFile.model_validate(
        json.loads(jobs_path.read_text(encoding="utf-8"))
    )
    cvs_payload = CVCorpusFile.model_validate(
        json.loads(cvs_path.read_text(encoding="utf-8"))
    )
    prefs_raw = json.loads(preferences_path.read_text(encoding="utf-8"))
    preferences = CandidatePreferences.model_validate(prefs_raw)
    cv = next((c for c in cvs_payload.cvs if c.id == cv_id), None)
    if cv is None:
        typer.echo(
            f"CV id {cv_id!r} not found. Available: {[c.id for c in cvs_payload.cvs]}",
            err=True,
        )
        raise typer.Exit(code=1)
    result = match_cv_to_jobs(cv, jobs_payload.jobs, preferences)
    typer.echo(dump_match_json(result))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
