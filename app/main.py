import os

import streamlit as st
from dotenv import load_dotenv

from app.cases import list_cases, load_case
from app.pipeline import ReviewPipeline
from app.schemas import CriterionStatus, Recommendation

load_dotenv()

ACCENT = "#2a6f97"  # calm blue — used for Run button + active step
ACCENT_HOVER = "#1d5678"

STATUS_COLORS = {
    CriterionStatus.MET: "#1b7f3b",
    CriterionStatus.NOT_MET: "#c0392b",
    CriterionStatus.NOT_DOCUMENTED: "#b7950b",
    CriterionStatus.NOT_APPLICABLE: "#5d6d7e",
    CriterionStatus.UNCLEAR: "#7d3c98",
}

STATUS_LABELS = {
    CriterionStatus.MET: "✅ MET",
    CriterionStatus.NOT_MET: "❌ NOT_MET",
    CriterionStatus.NOT_DOCUMENTED: "⚠️ NOT_DOCUMENTED",
    CriterionStatus.NOT_APPLICABLE: "⏸️ NOT_APPLICABLE",
    CriterionStatus.UNCLEAR: "❓ UNCLEAR",
}

RECOMMENDATION_COLORS = {
    Recommendation.LIKELY_MEETS: "#1b7f3b",
    Recommendation.NEEDS_MORE_DOCUMENTATION: "#b7950b",
    Recommendation.LIKELY_DOES_NOT_MEET: "#c0392b",
    Recommendation.URGENT_HUMAN_REVIEW: "#922b21",
}

RECOMMENDATION_GLOSS = {
    Recommendation.LIKELY_MEETS: "Documentation appears to support medical necessity under the retrieved policy pathway.",
    Recommendation.NEEDS_MORE_DOCUMENTATION: "Required evidence is incomplete — request additional documentation before deciding.",
    Recommendation.LIKELY_DOES_NOT_MEET: "Evidence is present but does not satisfy the policy criteria.",
    Recommendation.URGENT_HUMAN_REVIEW: "Red-flag findings detected — escalate for immediate clinical review.",
}


def main() -> None:
    st.set_page_config(
        page_title="Medical Necessity Evidence Gap Reviewer",
        page_icon="🩺",
        layout="wide",
    )
    _inject_theme_css()

    cases = list_cases()
    if not cases:
        st.error("No cases found in data/cases/.")
        return

    labels = {c.folder_name: f"{c.title} ({c.id})" for c in cases}

    st.title("Medical Necessity Evidence Gap Reviewer")
    st.caption(
        "Prior authorization decision support prototype — synthetic cases only. "
        "Not for clinical use."
    )
    if os.getenv("DEMO_MODE", "false").lower() == "true":
        st.info("DEMO_MODE on — using pre-authored mock reviews (no API key required).")

    selected_folder = st.selectbox(
        "Select case",
        options=list(labels.keys()),
        format_func=lambda key: labels[key],
    )

    if st.session_state.get("result_folder") != selected_folder:
        st.session_state.pop("result", None)
        st.session_state["result_folder"] = selected_folder
        st.session_state["active_step"] = "1"

    # Cache the loaded case (and its RAG retrieval) per folder so switching
    # steps in the sidebar doesn't re-run query generation / vector search.
    if st.session_state.get("case_request_folder") != selected_folder:
        st.session_state["case_request"] = load_case(selected_folder)
        st.session_state["case_request_folder"] = selected_folder
    case_request = st.session_state["case_request"]

    st.markdown(f"**Requested service:** {case_request.requested_service}")
    if case_request.retrieval_plan is not None:
        st.markdown(
            f"**Detected service slug:** `{case_request.retrieval_plan.service_slug}` · "
            f"**Domain filter:** `{case_request.retrieval_plan.domain_filter}`"
        )

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

    if st.button("Run agentic review", type="primary", use_container_width=False):
        with st.spinner(
            "Step 1 Retrieval → Step 2 Decompose → Step 3 Extract → "
            "Step 4 Match → Step 5 Governance..."
        ):
            try:
                pipeline = ReviewPipeline()
                result = pipeline.run(case_request, case_folder=selected_folder)
                st.session_state["result"] = result
                st.session_state["result_folder"] = selected_folder
                st.session_state["active_step"] = "1"
            except Exception as exc:
                st.error(f"Review failed: {exc}")
                st.info(
                    "Tip: set `OPENAI_API_KEY` in `.env`, or use `DEMO_MODE=true` "
                    "to run without an API key."
                )

    result = st.session_state.get("result")
    _render_sidebar_steps(review_complete=result is not None)

    if result is None:
        st.divider()
        st.caption("Run the agentic review to see the recommendation for this case.")
        return

    review = result.review

    st.divider()
    st.markdown("### Recommendation summary")
    st.caption("Overall outcome for this case — independent of the step view below.")
    _render_recommendation(review)
    _render_summary_metrics(review, result.governance)

    st.divider()
    active_step = st.session_state.get("active_step", "1")
    step_meta = {num: (title, desc) for num, title, desc in PIPELINE_STEPS}
    step_title, step_desc = step_meta.get(active_step, ("", ""))
    st.markdown(f"### Pipeline detail · Step {active_step}: {step_title}")
    st.caption(step_desc)

    if active_step == "1":
        _render_retrieved_chunks(case_request)
    elif active_step == "2":
        _render_decompose_step(review)
    elif active_step == "3":
        _render_extract_step(review)
    elif active_step == "4":
        _render_criteria_table(review)
        _render_gaps_and_next_steps(review)
    else:
        _render_governance(result.governance)


def _inject_theme_css() -> None:
    st.markdown(
        f"""
        <style>
        /* Primary action + active step — blue accent instead of Streamlit red */
        button[data-testid="baseButton-primary"],
        button[kind="primary"] {{
            background-color: {ACCENT} !important;
            border-color: {ACCENT} !important;
            color: #ffffff !important;
        }}
        button[data-testid="baseButton-primary"]:hover,
        button[kind="primary"]:hover {{
            background-color: {ACCENT_HOVER} !important;
            border-color: {ACCENT_HOVER} !important;
            color: #ffffff !important;
        }}
        section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {{
            opacity: 0.4;
        }}
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {{
            opacity: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


PIPELINE_STEPS = [
    ("1", "Retrieval", "Generate search queries and retrieve relevant policy chunks."),
    ("2", "Decompose", "Turn retrieved policy text into a criteria checklist."),
    ("3", "Extract", "Pull documented patient facts from the clinical note."),
    ("4", "Match", "Compare facts to criteria; surface gaps and recommendation."),
    ("5", "Governance", "Verify audit fields: policy source, evidence, status, explanation."),
]


def _render_sidebar_steps(*, review_complete: bool) -> None:
    if "active_step" not in st.session_state:
        st.session_state["active_step"] = "1"

    with st.sidebar:
        st.header("Pipeline steps")
        if review_complete:
            st.caption("Click a step to view its output. Other steps are dimmed.")
        else:
            st.caption("Run the review to unlock these steps.")

        for number, title, description in PIPELINE_STEPS:
            is_active = st.session_state["active_step"] == number and review_complete
            opacity = "1" if is_active else "0.35"

            if st.button(
                f"{'●' if is_active else '○'}  Step {number} · {title}",
                key=f"nav_step_{number}",
                disabled=not review_complete,
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state["active_step"] = number
                st.rerun()

            st.markdown(
                f'<div style="opacity:{opacity}; font-size:0.8rem; color:#b0b3b8;'
                f'margin:-0.25rem 0 0.85rem 0.15rem;">{description}</div>',
                unsafe_allow_html=True,
            )


def _render_recommendation(review) -> None:
    color = RECOMMENDATION_COLORS.get(review.recommendation, "#333")
    gloss = RECOMMENDATION_GLOSS.get(review.recommendation, "")
    st.markdown(
        f"""
        <div style="padding: 1.1rem 1.25rem; border-radius: 10px; border: 2px solid {color};
        background: {color}14; margin-bottom: 0.75rem;">
        <div style="font-size:0.8rem; letter-spacing:0.04em; text-transform:uppercase;
        color:{color}; font-weight:600; margin-bottom:0.35rem;">Recommendation</div>
        <h2 style="margin:0; color:{color}; font-size:1.55rem;">
        {review.recommendation.value}</h2>
        <p style="margin:0.55rem 0 0 0; color:#e8eaed;">{gloss}</p>
        <p style="margin:0.65rem 0 0 0; font-size:0.95rem; color:#e8eaed;">
        <strong>Confidence:</strong> {review.confidence.value}
        &nbsp;·&nbsp;
        <strong>Uncertainty:</strong> {"Yes" if review.uncertainty_flag else "No"}
        &nbsp;·&nbsp;
        <strong>Pathway:</strong> {review.pathway_applied or "None identified"}
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(review.rationale)


def _render_summary_metrics(review, governance) -> None:
    met = sum(1 for row in review.criteria_evaluation if row.status == CriterionStatus.MET)
    gaps = len(review.evidence_gaps)
    unclear = sum(
        1 for row in review.criteria_evaluation if row.status == CriterionStatus.UNCLEAR
    )
    gov_label = "Passed" if governance.passed else f"{len(governance.violations)} issue(s)"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Criteria met", f"{met}/{len(review.criteria_evaluation)}")
    m2.metric("Evidence gaps", gaps)
    m3.metric("Unclear criteria", unclear)
    m4.metric("Governance", gov_label)


def _render_governance(governance) -> None:
    st.subheader("Step 5 · Governance check")
    if governance.passed:
        st.success("All governance checks passed.")
    else:
        st.warning(f"{len(governance.violations)} governance issue(s) found.")
        for violation in governance.violations:
            label = violation.criterion_id or "review"
            st.markdown(f"- **{violation.rule_id}** ({label}): {violation.message}")


def _render_decompose_step(review) -> None:
    st.subheader("Step 2 · Decomposed policy criteria")
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


def _render_extract_step(review) -> None:
    st.subheader("Step 3 · Extracted patient facts")
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
    st.subheader("Step 1 · Retrieved policy snippets (RAG)")
    if case_request.retrieval_plan is not None:
        st.markdown("**Retrieval queries**")
        for query in case_request.retrieval_plan.queries:
            st.markdown(f"- {query}")

    if not case_request.retrieved_chunks:
        st.info("No retrieved chunks found. Falling back to the full synthetic policy file.")
        return

    for chunk in case_request.retrieved_chunks:
        with st.container(border=True):
            st.markdown(f"**{chunk.section_title}**")
            st.caption(
                f"`{chunk.chunk_id}` · `{chunk.service}` · score `{chunk.score}`"
            )
            st.write(chunk.text)


def _status_badge_html(status: CriterionStatus) -> str:
    color = STATUS_COLORS[status]
    label = STATUS_LABELS[status]
    return (
        f'<span style="display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;'
        f'background:{color}22;color:{color};font-weight:600;font-size:0.85rem;'
        f'white-space:nowrap;">{label}</span>'
    )


def _render_criteria_table(review) -> None:
    st.subheader("Step 4 · Criteria matching table")
    if not review.criteria_evaluation:
        st.write("No criteria evaluation returned.")
        return

    # Color-coded HTML table so status badges render clearly in the demo.
    header = (
        "<tr>"
        "<th style='text-align:left;padding:0.55rem;'>Criterion</th>"
        "<th style='text-align:left;padding:0.55rem;'>Status</th>"
        "<th style='text-align:left;padding:0.55rem;'>Policy source</th>"
        "<th style='text-align:left;padding:0.55rem;'>Patient evidence</th>"
        "<th style='text-align:left;padding:0.55rem;'>Explanation</th>"
        "</tr>"
    )
    body_rows = []
    for row in review.criteria_evaluation:
        color = STATUS_COLORS[row.status]
        body_rows.append(
            "<tr style='border-top:1px solid #e6e6e6;'>"
            f"<td style='padding:0.6rem;vertical-align:top;border-left:4px solid {color};'>"
            f"{row.criterion_text}</td>"
            f"<td style='padding:0.6rem;vertical-align:top;'>{_status_badge_html(row.status)}</td>"
            f"<td style='padding:0.6rem;vertical-align:top;'>{row.policy_source}</td>"
            f"<td style='padding:0.6rem;vertical-align:top;'>{row.patient_evidence}</td>"
            f"<td style='padding:0.6rem;vertical-align:top;'>{row.explanation}</td>"
            "</tr>"
        )

    st.markdown(
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.92rem;'>"
        f"<thead>{header}</thead><tbody>{''.join(body_rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


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
