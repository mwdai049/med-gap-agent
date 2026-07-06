import json
import os
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from app.cases import load_mock_review
from app.governance import apply_uncertainty_flag, validate_review
from app.prompts import (
    DECOMPOSE_POLICY_SYSTEM,
    DECOMPOSE_POLICY_USER,
    EXTRACT_FACTS_SYSTEM,
    EXTRACT_FACTS_USER,
    FINAL_REVIEW_SYSTEM,
    FINAL_REVIEW_USER,
)
from app.schemas import (
    DecomposedPolicy,
    ExtractedFacts,
    MedicalNecessityReview,
    PipelineResult,
    ReviewRequest,
)

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Near-zero temperature for extraction and classification steps
TEMP_EXTRACTION = float(os.getenv("OPENAI_TEMPERATURE_EXTRACTION", "0"))
TEMP_CLASSIFICATION = float(os.getenv("OPENAI_TEMPERATURE_CLASSIFICATION", "0"))


class ReviewPipeline:
    def __init__(self) -> None:
        self.demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        self.client = OpenAI() if not self.demo_mode else None

    def run(self, request: ReviewRequest, case_folder: str | None = None) -> PipelineResult:
        if self.demo_mode:
            if not case_folder:
                raise ValueError("case_folder is required in DEMO_MODE")
            review = load_mock_review(case_folder)
        else:
            review = self._run_llm(request)

        review = apply_uncertainty_flag(review)
        governance = validate_review(review)
        return PipelineResult(review=review, governance=governance)

    def _run_llm(self, request: ReviewRequest) -> MedicalNecessityReview:
        decomposed = self._parse(
            DecomposedPolicy,
            DECOMPOSE_POLICY_SYSTEM,
            DECOMPOSE_POLICY_USER.format(
                requested_service=request.requested_service,
                policy_text=request.policy_text,
            ),
            temperature=TEMP_EXTRACTION,
        )
        facts = self._parse(
            ExtractedFacts,
            EXTRACT_FACTS_SYSTEM,
            EXTRACT_FACTS_USER.format(
                requested_service=request.requested_service,
                chart_text=request.chart_text,
            ),
            temperature=TEMP_EXTRACTION,
        )
        review = self._parse(
            MedicalNecessityReview,
            FINAL_REVIEW_SYSTEM,
            FINAL_REVIEW_USER.format(
                requested_service=request.requested_service,
                policy_text=request.policy_text,
                chart_text=request.chart_text,
                decomposed_criteria=json.dumps(
                    [c.model_dump() for c in decomposed.criteria], indent=2
                ),
                extracted_facts=json.dumps([f.model_dump() for f in facts.facts], indent=2),
            ),
            temperature=TEMP_CLASSIFICATION,
        )
        review.decomposed_criteria = decomposed.criteria
        review.extracted_facts = facts.facts
        return review

    def _parse(
        self,
        model_type: type[T],
        system: str,
        user: str,
        *,
        temperature: float,
    ) -> T:
        if self.client is None:
            raise RuntimeError("OpenAI client is not configured")

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=model_type,
            temperature=temperature,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(f"Model returned no parsed output for {model_type.__name__}")
        return parsed
