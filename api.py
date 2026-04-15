"""FastAPI service for the online matching pipeline.

POST /match
  - multipart: cv (PDF file)
  - form fields (all optional): model, role_type, remote, min_salary,
                                 seniority, company_stage, company_size, location
  -> RankedMatchResult JSON
"""

import os
import json
import tempfile
from typing import Optional

from fastapi import File, Form, FastAPI, UploadFile, HTTPException
from fastapi.responses import Response

from matchmaker.vocab import (
    RoleType,
    CompanySize,
    CompanyStage,
    SeniorityLevel,
    RemotePreference,
)
from matchmaker.data_models import RankedMatchResult, CandidatePreferences
from matchmaker.online_pipeline.pipeline import OnlinePipeline, _parse_enum_list

app = FastAPI(
    title="Match Maker",
    description="CV–job matcher: upload a CV PDF and receive ranked job matches.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/match", response_model=RankedMatchResult)
async def match(
    cv: UploadFile = File(..., description="CV in PDF format."),
    model: str = Form("gpt-4o-mini", description="LLM model name."),
    api_key: Optional[str] = Form(
        None, description="OpenAI API key (overrides env var)."
    ),
    role_type: list[str] = Form([], description="Preferred role types."),
    remote: str = Form("no_preference", description="Remote preference."),
    min_salary: Optional[int] = Form(None, description="Minimum salary (GBP)."),
    seniority: list[str] = Form([], description="Preferred seniority levels."),
    company_stage: list[str] = Form([], description="Preferred company stages."),
    company_size: list[str] = Form([], description="Preferred company sizes."),
    location: list[str] = Form([], description="Preferred locations (partial match)."),
) -> Response:
    if cv.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    pdf_bytes = await cv.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    has_prefs = any(
        [
            role_type,
            remote != "no_preference",
            min_salary,
            seniority,
            company_stage,
            company_size,
            location,
        ]
    )
    prefs: CandidatePreferences | None = None
    if has_prefs:
        prefs = CandidatePreferences(
            preferred_role_types=_parse_enum_list(role_type, RoleType),
            remote_preference=RemotePreference(remote),
            min_salary=min_salary,
            preferred_seniority=_parse_enum_list(seniority, SeniorityLevel),
            preferred_company_stages=_parse_enum_list(company_stage, CompanyStage),
            preferred_company_sizes=_parse_enum_list(company_size, CompanySize),
            preferred_locations=location,
        )

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or None
    pipeline = OnlinePipeline(model=model, api_key=resolved_key, verbose=False)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        result: RankedMatchResult = await pipeline.match(
            tmp_path, preferences=prefs, print_results=False
        )
    finally:
        os.unlink(tmp_path)

    return Response(
        content=json.dumps(result.model_dump(mode="json"), indent=2),
        media_type="application/json",
    )
