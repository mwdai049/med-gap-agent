import os

import streamlit as st
from dotenv import load_dotenv

from app.cases import list_cases, load_case
from app.pipeline import ReviewPipeline
from app.schemas import CriterionStatus, Recommendation

load_dotenv()

STATUS_COLORS = {
    CriterionStatus.MET: "#1b7f3b",
    CriterionStatus.NOT_MET: "#c0392b",
    CriterionStatus.NOT_DOCUMENTED: "#b7950b",
    CriterionStatus.NOT_APPLICABLE: "#5d6d7e",
    CriterionStatus.UNCLEAR: "#7d3c98",
}

RECOMMENDATION_COLORS = {
    Recommendation.LIKELY_MEETS: "#1b7f3b",
    Recommendation.NEEDS_MORE_DOCUMENTATION: "#b7950b",
    Recommendation.LIKELY_DOES_NOT_MEET: "#c0392b",
    Recommendation.URGENT_HUMAN_REVIEW: "#922b21",
}


def main() -> None:
    st.set_page_config(
        page_title="Medical Necessity Evidence Gap Reviewer",
        page_icon="🩺",
        layout="wide",
    )

    demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
    st.title("Medical Necessity Evidence Gap Reviewer")
    st.caption(
        "Prior authorization decision support prototype — synthetic cases only. "
        "Not for clinical use."
    )

    if demo_mode:
        st.info("Running in **DEMO_MODE** with pre-authored mock responses (no API key required).")

    cases = list_cases()
    if not cases:
        st.error("No cases found in data/cases/.")
        return

    labels = {c.folder_name: f"{c.title} ({c.id})" for c in cases}
    selected_folder = st.selectbox(
        "Select case",
        options=list(labels.keys()),
        format_func=lambda key: labels[key],
    )

    case_request = load_case(selected_folder)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Patient note")
        st.text_area(
            "Chart",
            value=case_request.chart_text,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )
    with col_right:
        st.subheader("Retrieved policy context")
        st.text_area(
            "Policy",
            value=case_request.policy_text,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    st.markdown(f"**Requested service:** {case_request.requested_service}")
    if case_request.retrieval_plan is not None:
        st.markdown(
            f"**Detected service slug:** `{case_request.retrieval_plan.service_slug}` | "
            f"**Domain filter:** `{case_request.retrieval_plan.domain_filter}`"
        )
    _render_retrieved_chunks(case_request)

    if st.button("Run agentic review", type="primary"):
        with st.spinner(
            "Decomposing policy → extracting facts → matching criteria → governance check..."
        ):
            try:
                pipeline = ReviewPipeline()
                result = pipeline.run(case_request, case_folder=selected_folder)
                st.session_state["result"] = result
            except Exception as exc:
                st.error(f"Review failed: {exc}")
                st.info(
                    "Tip: set `OPENAI_API_KEY` in `.env`, or use `DEMO_MODE=true` "
                    "to run without an API key."
                )

    result = st.session_state.get("result")
    if result is None:
        return

    review = result.review
    _render_recommendation(review)
    _render_governance(result.governance)
    _render_intermediate_steps(review)
    _render_criteria_table(review)
    _render_gaps_and_next_steps(review)


def _render_recommendation(review) -> None:
    color = RECOMMENDATION_COLORS.get(review.recommendation, "#333")
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 8px; border: 2px solid {color};
        background: {color}15;">
        <h3 style="margin:0; color:{color};">Recommendation: {review.recommendation.value}</h3>
        <p style="margin:0.5rem 0 0 0;"><strong>Confidence:</strong> {review.confidence.value}</p>
        <p style="margin:0.25rem 0 0 0;"><strong>Uncertainty flag:</strong>
        {"Yes" if review.uncertainty_flag else "No"}</p>
        <p style="margin:0.25rem 0 0 0;"><strong>Pathway:</strong>
        {review.pathway_applied or "None identified"}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(review.rationale)


def _render_governance(governance) -> None:
    with st.expander("Step 4: Governance check", expanded=True):
        if governance.passed:
            st.success("All governance checks passed.")
        else:
            st.warning(f"{len(governance.violations)} governance issue(s) found.")
            for violation in governance.violations:
                label = violation.criterion_id or "review"
                st.markdown(f"- **{violation.rule_id}** ({label}): {violation.message}")


def _render_intermediate_steps(review) -> None:
    with st.expander("Step 1: Decomposed policy criteria", expanded=False):
        if review.decomposed_criteria:
            st.dataframe(
                [
                    {
                        "ID": c.criterion_id,
                        "Pathway": c.pathway,
                        "Criterion": c.criterion_text,
                    }
                    for c in review.decomposed_criteria
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No decomposed criteria returned.")

    with st.expander("Step 2: Extracted patient facts", expanded=False):
        if review.extracted_facts:
            st.dataframe(
                [
                    {
                        "ID": f.fact_id,
                        "Fact": f.fact_text,
                        "Quote": f.source_quote or "",
                    }
                    for f in review.extracted_facts
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No facts extracted.")


def _render_retrieved_chunks(case_request) -> None:
    with st.expander("Retrieved policy snippets (RAG)", expanded=True):
        if case_request.retrieval_plan is not None:
            st.markdown("**Retrieval queries**")
            for query in case_request.retrieval_plan.queries:
                st.markdown(f"- {query}")

        if not case_request.retrieved_chunks:
            st.info("No retrieved chunks found. Falling back to the full synthetic policy file.")
            return

        for chunk in case_request.retrieved_chunks:
            st.markdown(
                f"**{chunk.section_title}**  \n"
                f"`{chunk.chunk_id}` | `{chunk.source_file}` | "
                f"`{chunk.service}` | score `{chunk.score}`"
            )
            st.caption(chunk.text)


def _render_criteria_table(review) -> None:
    st.subheader("Criteria matching table")
    rows = []
    for row in review.criteria_evaluation:
        rows.append(
            {
                "Criterion": row.criterion_text,
                "Status": row.status.value,
                "Policy source": row.policy_source,
                "Patient evidence": row.patient_evidence,
                "Explanation": row.explanation,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_gaps_and_next_steps(review) -> None:
    col_gaps, col_steps = st.columns(2)

    with col_gaps:
        st.subheader("Evidence gaps")
        if review.evidence_gaps:
            for gap in review.evidence_gaps:
                st.markdown(f"- {gap.description}")
        else:
            st.write("No evidence gaps identified.")

    with col_steps:
        st.subheader("Suggested next steps")
        if review.suggested_next_steps:
            for step in review.suggested_next_steps:
                st.markdown(f"- {step}")
        else:
            st.write("No additional steps suggested.")

    if review.policy_citations:
        st.subheader("Policy citations")
        for citation in review.policy_citations:
            st.markdown(f"- {citation}")


if __name__ == "__main__":
    main()
