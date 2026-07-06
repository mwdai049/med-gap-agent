from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CriterionStatus(str, Enum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    NOT_DOCUMENTED = "NOT_DOCUMENTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNCLEAR = "UNCLEAR"


class Recommendation(str, Enum):
    LIKELY_MEETS = "LIKELY_MEETS"
    NEEDS_MORE_DOCUMENTATION = "NEEDS_MORE_DOCUMENTATION"
    LIKELY_DOES_NOT_MEET = "LIKELY_DOES_NOT_MEET"
    URGENT_HUMAN_REVIEW = "URGENT_HUMAN_REVIEW"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PolicyCriterion(BaseModel):
    criterion_id: str
    criterion_text: str
    pathway: str


class RetrievedPolicyChunk(BaseModel):
    chunk_id: str
    source_file: str
    section_title: str
    text: str
    score: float
    domain: str
    service: str
    pathway: str
    heading_level: int


class ExtractedFact(BaseModel):
    fact_id: str
    fact_text: str
    source_quote: str | None = None


class CriterionEvaluation(BaseModel):
    criterion_id: str
    criterion_text: str
    pathway: str | None = None
    status: CriterionStatus
    policy_source: str = Field(
        description="Policy section or pathway this criterion maps to, e.g. Pathway 1"
    )
    patient_evidence: str = Field(
        description="Supporting quote or summary from the patient note; state if not documented"
    )
    explanation: str = Field(
        description="Brief reason this status was assigned for this criterion"
    )
    source: str = "Patient note"


class EvidenceGap(BaseModel):
    description: str
    related_criterion_id: str | None = None


class DecomposedPolicy(BaseModel):
    criteria: list[PolicyCriterion]


class ExtractedFacts(BaseModel):
    facts: list[ExtractedFact]


class MedicalNecessityReview(BaseModel):
    recommendation: Recommendation
    confidence: Confidence
    uncertainty_flag: bool = Field(
        description="True when confidence is low or any criterion status is UNCLEAR"
    )
    pathway_applied: str | None = None
    decomposed_criteria: list[PolicyCriterion] = Field(default_factory=list)
    extracted_facts: list[ExtractedFact]
    criteria_evaluation: list[CriterionEvaluation]
    evidence_gaps: list[EvidenceGap]
    rationale: str
    suggested_next_steps: list[str] = Field(default_factory=list)
    policy_citations: list[str] = Field(default_factory=list)


class GovernanceViolation(BaseModel):
    rule_id: str
    message: str
    criterion_id: str | None = None


class GovernanceReport(BaseModel):
    passed: bool
    violations: list[GovernanceViolation] = Field(default_factory=list)


class PipelineResult(BaseModel):
    review: MedicalNecessityReview
    governance: GovernanceReport


class RetrievalPlan(BaseModel):
    requested_service: str
    service_slug: str
    domain_filter: Literal["radiology_prior_auth", "all"]
    queries: list[str] = Field(min_length=1, max_length=3)


class ReviewRequest(BaseModel):
    case_id: str
    case_title: str
    requested_service: str
    policy_text: str
    chart_text: str
    retrieved_chunks: list[RetrievedPolicyChunk] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan | None = None
