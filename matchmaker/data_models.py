"""Pydantic models for jobs, CVs, preferences, and match output."""

from datetime import datetime, timezone

from pydantic import Field, BaseModel, field_validator

from matchmaker.vocab import (
    RoleType,
    CompanySize,
    CompanyStage,
    RemotePolicy,
    SeniorityLevel,
    DomainExpertise,
    RemotePreference,
)

_SYNONYMS: dict[str, str] = {
    "finance": DomainExpertise.fintech.value,
    "banking": DomainExpertise.credit_risk.value,
    "ads": DomainExpertise.adtech.value,
    "local_government": DomainExpertise.public_sector.value,
}


def _validate_domain_tags(values: list[str]) -> list[str]:
    allowed = {d.value for d in DomainExpertise}
    out: list[str] = []
    for v in values:
        raw = v.strip()
        key = raw.lower().replace(" ", "_").replace("-", "_")
        if raw in allowed:
            out.append(raw)
        elif key in allowed:
            out.append(key)
        elif key in _SYNONYMS:
            out.append(_SYNONYMS[key])
        else:
            out.append(DomainExpertise.general_ml_ai.value)
    seen: set[str] = set()
    unique: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return unique


class LeadershipRequiredJob(BaseModel):
    required: bool = False
    team_size_hint: int | None = None


class HasLeadershipExperience(BaseModel):
    has: bool = False
    max_team_size_managed: int | None = None


class OpenSourceContributions(BaseModel):
    has: bool = False
    note: str | None = None


class JobRecord(BaseModel):
    id: str = Field(..., description="Stable id, e.g. slug from filename")
    title: str
    company: str
    industry: str = Field(..., description="Inferred sector label")
    location: str
    remote_policy: RemotePolicy
    days_in_office_per_week: int | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    seniority_level: SeniorityLevel
    role_type: RoleType
    company_stage: CompanyStage
    company_size: CompanySize
    key_responsibilities: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    domain_expertise_required: list[str] = Field(default_factory=list)
    leadership_required: LeadershipRequiredJob = Field(
        default_factory=LeadershipRequiredJob
    )
    visa_sponsorship: bool = False
    source_file: str | None = None
    extracted_at: str | None = None

    @field_validator("domain_expertise_required", mode="before")
    @classmethod
    def normalize_domains(cls, v: list[str] | None) -> list[str]:
        if not v:
            return []
        return _validate_domain_tags(list(v))


class CVRecord(BaseModel):
    id: str
    name: str
    location: str
    total_years_experience: int = Field(ge=0)
    most_recent_job_title: str
    seniority_level: SeniorityLevel
    primary_role_type: RoleType
    technical_skills: list[str] = Field(default_factory=list)
    domain_expertise: list[str] = Field(default_factory=list)
    industries_worked_in: list[str] = Field(default_factory=list)
    career_highlights: list[str] = Field(default_factory=list, max_length=10)
    has_leadership_experience: HasLeadershipExperience = Field(
        default_factory=HasLeadershipExperience
    )
    has_phd: bool = False
    open_source_contributions: OpenSourceContributions = Field(
        default_factory=OpenSourceContributions
    )
    source_file: str | None = None
    extracted_at: str | None = None

    @field_validator("domain_expertise", mode="before")
    @classmethod
    def normalize_cv_domains(cls, v: list[str] | None) -> list[str]:
        if not v:
            return []
        return _validate_domain_tags(list(v))


class CandidatePreferences(BaseModel):
    """Query-time preferences — not part of CV JSON."""

    preferred_role_types: list[RoleType] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    remote_preference: RemotePreference = RemotePreference.no_preference
    min_salary: int | None = None


class RankedJob(BaseModel):
    job_id: str
    score: float
    rank: int
    rationale: str


class MatchSummary(BaseModel):
    best_fit_job_id: str
    overall_comment: str


class RankedMatchResult(BaseModel):
    ranked_jobs: list[RankedJob]
    summary: MatchSummary


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
