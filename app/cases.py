import json
from pathlib import Path

from pydantic import BaseModel

from app.rag import build_policy_context, retrieve_policy_chunks
from app.schemas import MedicalNecessityReview, ReviewRequest


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CASES_DIR = DATA_DIR / "cases"
POLICIES_DIR = DATA_DIR / "policies"


class CaseSummary(BaseModel):
    id: str
    title: str
    requested_service: str
    folder_name: str


def list_cases() -> list[CaseSummary]:
    cases: list[CaseSummary] = []
    for case_dir in sorted(CASES_DIR.iterdir()):
        if not case_dir.is_dir():
            continue
        request_path = case_dir / "request.json"
        if not request_path.exists():
            continue
        request_data = json.loads(request_path.read_text(encoding="utf-8"))
        cases.append(
            CaseSummary(
                id=request_data["id"],
                title=request_data["title"],
                requested_service=request_data["requested_service"],
                folder_name=case_dir.name,
            )
        )
    return cases


def load_case(folder_name: str) -> ReviewRequest:
    case_dir = CASES_DIR / folder_name
    request_data = json.loads((case_dir / "request.json").read_text(encoding="utf-8"))
    chart_text = (case_dir / "chart.txt").read_text(encoding="utf-8").strip()
    top_k = request_data.get("retrieval_top_k", 3)
    retrieval_plan, retrieved_chunks = retrieve_policy_chunks(
        request_data["requested_service"],
        chart_text,
        top_k=top_k,
    )

    if retrieved_chunks:
        policy_text = build_policy_context(retrieved_chunks)
    else:
        policy_text = load_policy(request_data["policy_file"])

    return ReviewRequest(
        case_id=request_data["id"],
        case_title=request_data["title"],
        requested_service=request_data["requested_service"],
        policy_text=policy_text,
        chart_text=chart_text,
        retrieved_chunks=retrieved_chunks,
        retrieval_plan=retrieval_plan,
    )


def load_policy(policy_file: str) -> str:
    return (POLICIES_DIR / policy_file).read_text(encoding="utf-8")


def load_mock_review(folder_name: str) -> MedicalNecessityReview:
    path = CASES_DIR / folder_name / "mock_review.json"
    return MedicalNecessityReview.model_validate_json(path.read_text(encoding="utf-8"))
