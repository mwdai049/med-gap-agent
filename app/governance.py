"""Deterministic governance checks — no LLM, audit-friendly."""

from app.schemas import (
    Confidence,
    CriterionStatus,
    GovernanceReport,
    GovernanceViolation,
    MedicalNecessityReview,
)


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


def validate_review(review: MedicalNecessityReview) -> GovernanceReport:
    violations: list[GovernanceViolation] = []

    if not review.policy_citations:
        violations.append(
            GovernanceViolation(
                rule_id="REVIEW_POLICY_CITATIONS",
                message="Recommendation must cite at least one policy section.",
            )
        )

    if _is_blank(review.rationale):
        violations.append(
            GovernanceViolation(
                rule_id="REVIEW_RATIONALE",
                message="Recommendation must include a rationale explanation.",
            )
        )

    if review.confidence not in Confidence:
        violations.append(
            GovernanceViolation(
                rule_id="REVIEW_UNCERTAINTY",
                message="Recommendation must include a confidence level (low, medium, high).",
            )
        )

    if review.uncertainty_flag and review.confidence == Confidence.HIGH:
        unclear_ids = [
            row.criterion_id
            for row in review.criteria_evaluation
            if row.status == CriterionStatus.UNCLEAR
        ]
        if not unclear_ids:
            violations.append(
                GovernanceViolation(
                    rule_id="REVIEW_UNCERTAINTY_FLAG",
                    message="uncertainty_flag is true but confidence is high with no UNCLEAR criteria.",
                )
            )

    if not review.criteria_evaluation:
        violations.append(
            GovernanceViolation(
                rule_id="CRITERIA_PRESENT",
                message="At least one criterion evaluation row is required.",
            )
        )

    for row in review.criteria_evaluation:
        prefix = row.criterion_id

        if _is_blank(row.policy_source):
            violations.append(
                GovernanceViolation(
                    rule_id="CRITERION_POLICY_SOURCE",
                    message=f"{prefix}: missing policy_source.",
                    criterion_id=row.criterion_id,
                )
            )

        if _is_blank(row.patient_evidence):
            violations.append(
                GovernanceViolation(
                    rule_id="CRITERION_PATIENT_EVIDENCE",
                    message=f"{prefix}: missing patient_evidence.",
                    criterion_id=row.criterion_id,
                )
            )

        if row.status is None:
            violations.append(
                GovernanceViolation(
                    rule_id="CRITERION_STATUS",
                    message=f"{prefix}: missing status label.",
                    criterion_id=row.criterion_id,
                )
            )

        if _is_blank(row.explanation):
            violations.append(
                GovernanceViolation(
                    rule_id="CRITERION_EXPLANATION",
                    message=f"{prefix}: missing explanation.",
                    criterion_id=row.criterion_id,
                )
            )

    return GovernanceReport(passed=len(violations) == 0, violations=violations)


def apply_uncertainty_flag(review: MedicalNecessityReview) -> MedicalNecessityReview:
    has_unclear = any(
        row.status == CriterionStatus.UNCLEAR for row in review.criteria_evaluation
    )
    review.uncertainty_flag = review.confidence == Confidence.LOW or has_unclear
    return review
