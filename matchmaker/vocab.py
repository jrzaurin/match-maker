# NOTE: we could let the LLM generate freely. For this task, we will use a
# controlled vocabulary that one can expand as needed.

from enum import StrEnum


class RoleType(StrEnum):
    ml_engineer = "ml_engineer"
    data_scientist = "data_scientist"
    data_engineer = "data_engineer"
    data_analyst = "data_analyst"
    ml_manager = "ml_manager"


class RemotePolicy(StrEnum):
    fully_remote = "fully_remote"
    hybrid = "hybrid"
    on_site = "on_site"


class SeniorityLevel(StrEnum):
    junior = "junior"
    mid = "mid"
    senior = "senior"
    lead = "lead"
    director = "director"
    vp = "vp"


class CompanyStage(StrEnum):
    startup = "startup"
    scale_up = "scale_up"
    enterprise = "enterprise"
    public = "public"


class CompanySize(StrEnum):
    small = "small"  # 1–50
    medium = "medium"  # 51–500
    large = "large"  # 500+


class RemotePreference(StrEnum):
    fully_remote = "fully_remote"
    hybrid = "hybrid"
    no_preference = "no_preference"


class DomainExpertise(StrEnum):
    """Shared closed list for job domain_expertise_required and CV domain_expertise."""

    adtech = "adtech"
    credit_risk = "credit_risk"
    fraud_detection = "fraud_detection"
    recsys = "recsys"
    retail = "retail"
    ecommerce = "ecommerce"
    fintech = "fintech"
    proptech = "proptech"
    public_sector = "public_sector"
    healthcare = "healthcare"
    marketing_analytics = "marketing_analytics"
    energy_utilities = "energy_utilities"
    insurance = "insurance"
    general_ml_ai = "general_ml_ai"
    sports_betting = "sports_betting"
    embedded_finance = "embedded_finance"
    field_service = "field_service"
    telecoms = "telecoms"


def enum_appendix() -> str:
    """Human-readable block for LLM prompts (must stay in sync with enums above)."""
    lines = [
        "## Controlled vocabulary (use EXACT strings)",
        "",
        "remote_policy: " + " | ".join(p.value for p in RemotePolicy),
        "seniority_level: " + " | ".join(s.value for s in SeniorityLevel),
        "role_type / primary_role_type / preferred_role_types: "
        + " | ".join(r.value for r in RoleType),
        "company_stage: " + " | ".join(c.value for c in CompanyStage),
        "company_size: " + " | ".join(c.value for c in CompanySize),
        "remote_preference: " + " | ".join(r.value for r in RemotePreference),
        "domain_expertise / domain_expertise_required / preferred_industries (use domain tags): "
        + " | ".join(d.value for d in DomainExpertise),
        "",
        "For industries_worked_in on CVs, use the same domain tags where possible, or short "
        "industry labels like fintech, retail, consulting.",
    ]
    return "\n".join(lines)
