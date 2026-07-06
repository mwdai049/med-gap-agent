# Medical Necessity Evidence Gap Reviewer

Hackathon proof-of-concept for Cotiviti intern assessment: an agentic reviewer that retrieves relevant policy snippets from a headed Markdown policy, compares synthetic patient notes against lumbar spine MRI medical necessity criteria, and surfaces evidence gaps.

**Decision support only — not for clinical use.**

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Option A: Demo mode (no API key)

```bash
# In .env
DEMO_MODE=true
```

```bash
streamlit run streamlit_app.py
```

Uses pre-authored `mock_review.json` files for the three lumbar MRI cases.

### Option B: Live LLM mode

```bash
# In .env
DEMO_MODE=false
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-mini
```

```bash
streamlit run streamlit_app.py
```

## Demo cases

| Case | Expected recommendation |
|------|-------------------------|
| Case A — Likely meets criteria | `LIKELY_MEETS` |
| Case B — Evidence gaps | `NEEDS_MORE_DOCUMENTATION` |
| Case C — Red flag / urgent pathway | `URGENT_HUMAN_REVIEW` |

## Project structure

```
app/
  main.py        # Streamlit UI
  pipeline.py    # 3-step agentic review (decompose → extract → match)
  prompts.py     # LLM prompts
  schemas.py     # Pydantic models
  cases.py       # Load cases and retrieved policy context
  rag.py         # Header-based chunking, retrieval planning, and Chroma search
data/
  policies/
    lumbar_mri_policy.md  # Single headed Markdown policy source
  cases/                  # Synthetic charts + mock reviews
```

## Agent flow

1. **Indexing time** — load `data/policies/lumbar_mri_policy.md`, clean text, split by Markdown headings/pathways, attach metadata, create embeddings, and persist chunks in Chroma.
2. **Query time** — deterministically detect the requested service and build a guarded retrieval plan with 3 search queries.
3. **Policy retrieval (RAG)** — retrieve top policy chunks with metadata filtering and hybrid vector + keyword scoring.
4. **Policy decomposer** — retrieved policy text → criteria checklist.
5. **Fact extractor** — patient note → documented facts only.
6. **Gap analyzer** — criterion table + recommendation + gaps.
7. **Governance check** — verify auditable output fields are present.

## Retrieval design

- Policy content is stored as a single clean Markdown file with meaningful headings.
- Chunking is header-first; recursive splitting only happens if a section becomes too long.
- Every chunk carries metadata such as `domain`, `service`, `pathway`, and `heading_level`.
- Retrieval uses:
  - deterministic service detection
  - LLM-generated query expansion with guardrails
  - Chroma vector similarity
  - exact-term keyword bonuses as a fallback

## Limitations

- Synthetic data only; no real PHI
- Paraphrased policy inspired by public guidelines (cite sources in written report)
- Human reviewer required for all decisions
- LLM outputs may vary; use `DEMO_MODE` for a stable recording demo
- Demo mode falls back to deterministic query generation and keyword retrieval when no API key is present

## Other Assessment deliverables

Found in `deliverables/`: Word report, PowerPoint, MP4 video.