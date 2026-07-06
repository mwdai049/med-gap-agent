import hashlib
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from app.schemas import RetrievedPolicyChunk, RetrievalPlan

load_dotenv()

POLICY_MD_PATH = Path(__file__).resolve().parent.parent / "data" / "policies" / "lumbar_mri_policy.md"
CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chroma"
COLLECTION_NAME = "policy_chunks"
DOMAIN = "radiology_prior_auth"
SERVICE_ALIAS_MAP = {
    "lumbar spine mri without contrast": "lumbar_mri",
    "lumbar spine mri": "lumbar_mri",
    "lumbar mri": "lumbar_mri",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "for",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "with",
}
MAX_SECTION_CHARS = 900
KEYWORD_WEIGHT = 0.1
VECTOR_K = int(os.getenv("POLICY_VECTOR_K", "5"))
TOP_K = int(os.getenv("POLICY_RAG_TOP_K", "4"))
QUERY_MODEL = os.getenv("OPENAI_QUERY_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
QUERY_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE_QUERY", "0"))

_CHROMA_CLIENT: chromadb.ClientAPI | None = None
_QUERY_CLIENT: OpenAI | None = None

QUERY_SYSTEM = """You generate guarded retrieval plans for healthcare policy RAG.

Rules:
- Stay strictly within the requested service.
- Domain filter must be radiology_prior_auth unless the service is unknown.
- Produce exactly 3 retrieval queries.
- Prefer medical necessity, prior authorization, policy, imaging, neurologic, red flag, and conservative therapy terminology when relevant.
- Do not introduce unrelated services.
"""

QUERY_USER = """Requested service: {requested_service}
Service slug: {service_slug}
Patient note:
{chart_text}

Return a retrieval plan for policy search.
"""


def build_retrieval_plan(requested_service: str, chart_text: str) -> RetrievalPlan:
    service_slug = detect_service_slug(requested_service)
    if _can_use_llm():
        client = _get_query_client()
        response = client.beta.chat.completions.parse(
            model=QUERY_MODEL,
            messages=[
                {"role": "system", "content": QUERY_SYSTEM},
                {
                    "role": "user",
                    "content": QUERY_USER.format(
                        requested_service=requested_service,
                        service_slug=service_slug,
                        chart_text=chart_text,
                    ),
                },
            ],
            response_format=RetrievalPlan,
            temperature=QUERY_TEMPERATURE,
        )
        parsed = response.choices[0].message.parsed
        if parsed is not None:
            parsed.requested_service = requested_service
            parsed.service_slug = service_slug
            if parsed.domain_filter not in {"radiology_prior_auth", "all"}:
                parsed.domain_filter = "radiology_prior_auth"
            return parsed

    return _default_retrieval_plan(requested_service, service_slug, chart_text)


def retrieve_policy_chunks(
    requested_service: str,
    chart_text: str,
    *,
    top_k: int = TOP_K,
) -> tuple[RetrievalPlan, list[RetrievedPolicyChunk]]:
    plan = build_retrieval_plan(requested_service, chart_text)
    if not _can_use_llm():
        chunks = _keyword_retrieve(plan, top_k=top_k)
        return plan, chunks

    collection = _get_or_build_collection()
    scored: dict[str, tuple[dict, float]] = {}

    where = {"domain": plan.domain_filter}
    if plan.service_slug != "unknown":
        where = {"$and": [{"domain": plan.domain_filter}, {"service": plan.service_slug}]}

    for query in plan.queries:
        query_embedding = _embed_texts([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k, VECTOR_K),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for document, metadata, distance in zip(documents, metadatas, distances):
            chunk_id = str(metadata["chunk_id"])
            score = _hybrid_score(query, document, metadata, distance)
            current = scored.get(chunk_id)
            if current is None or score > current[1]:
                stored_metadata = dict(metadata)
                stored_metadata["text"] = document
                scored[chunk_id] = (stored_metadata, score)

    chunks = [
        RetrievedPolicyChunk(
            chunk_id=metadata["chunk_id"],
            source_file=metadata["source_file"],
            section_title=metadata["section_title"],
            text=metadata["text"],
            score=round(score, 3),
            domain=metadata["domain"],
            service=metadata["service"],
            pathway=metadata["pathway"],
            heading_level=int(metadata["heading_level"]),
        )
        for metadata, score in scored.values()
    ]
    chunks.sort(key=lambda chunk: chunk.score, reverse=True)
    return plan, chunks[:top_k]


def build_policy_context(chunks: list[RetrievedPolicyChunk]) -> str:
    sections: list[str] = []
    for chunk in chunks:
        sections.append(
            f"[{chunk.chunk_id}] {chunk.section_title}\n"
            f"Pathway: {chunk.pathway}\n"
            f"Domain: {chunk.domain} | Service: {chunk.service}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(sections)


def detect_service_slug(requested_service: str) -> str:
    normalized = re.sub(r"\s+", " ", requested_service.strip().lower())
    for key, slug in SERVICE_ALIAS_MAP.items():
        if key in normalized:
            return slug
    if "lumbar" in normalized and "mri" in normalized:
        return "lumbar_mri"
    return "unknown"


def _get_or_build_collection():
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = _CHROMA_CLIENT.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = _load_policy_chunks()
    expected_ids = [chunk["chunk_id"] for chunk in chunks]
    existing_ids = set(collection.get(include=[], where={"domain": DOMAIN}).get("ids", []))
    if set(expected_ids) != existing_ids:
        if existing_ids:
            collection.delete(ids=list(existing_ids))
        embeddings = _embed_texts([chunk["text"] for chunk in chunks])
        collection.add(
            ids=expected_ids,
            embeddings=embeddings,
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                    "section_title": chunk["section_title"],
                    "domain": chunk["domain"],
                    "service": chunk["service"],
                    "pathway": chunk["pathway"],
                    "heading_level": chunk["heading_level"],
                }
                for chunk in chunks
            ],
        )

    return collection


def _load_policy_chunks() -> list[dict]:
    markdown = _clean_markdown(POLICY_MD_PATH.read_text(encoding="utf-8"))
    sections = _split_markdown_by_headers(markdown, source_file=POLICY_MD_PATH.name)
    chunks: list[dict] = []
    for section in sections:
        chunks.extend(_split_large_section(section))
    return chunks


def _clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_markdown_by_headers(markdown: str, *, source_file: str) -> list[dict]:
    sections: list[dict] = []
    current_heading = None
    current_level = None
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("#"):
            if current_heading is not None:
                sections.append(
                    _build_section(
                        source_file=source_file,
                        heading=current_heading,
                        heading_level=current_level or 1,
                        body="\n".join(current_lines).strip(),
                    )
                )
            hashes, title = line.split(" ", 1)
            current_heading = title.strip()
            current_level = len(hashes)
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections.append(
            _build_section(
                source_file=source_file,
                heading=current_heading,
                heading_level=current_level or 1,
                body="\n".join(current_lines).strip(),
            )
        )
    return [
        section
        for section in sections
        if section["text"].strip() and int(section["heading_level"]) > 1
    ]


def _build_section(*, source_file: str, heading: str, heading_level: int, body: str) -> dict:
    pathway = heading
    service = "lumbar_mri" if "lumbar" in heading.lower() or "lumbar" in body.lower() else "general"
    chunk_id = _make_chunk_id(source_file, heading, body)
    text = f"{heading}\n\n{body}".strip()
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "section_title": heading,
        "text": text,
        "domain": DOMAIN,
        "service": service,
        "pathway": pathway,
        "heading_level": heading_level,
    }


def _split_large_section(section: dict) -> list[dict]:
    text = section["text"]
    if len(text) <= MAX_SECTION_CHARS:
        return [section]

    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[dict] = []
    buffer: list[str] = []
    for part in parts:
        candidate = "\n\n".join(buffer + [part]).strip()
        if buffer and len(candidate) > MAX_SECTION_CHARS:
            chunks.append(_section_part(section, buffer, len(chunks) + 1))
            buffer = [part]
        else:
            buffer.append(part)

    if buffer:
        chunks.append(_section_part(section, buffer, len(chunks) + 1))
    return chunks


def _section_part(section: dict, parts: list[str], index: int) -> dict:
    chunk = dict(section)
    chunk["chunk_id"] = f"{section['chunk_id']}_part_{index}"
    chunk["text"] = "\n\n".join(parts)
    return chunk


def _make_chunk_id(source_file: str, heading: str, body: str) -> str:
    digest = hashlib.md5(f"{source_file}:{heading}:{body}".encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
    return f"{slug}_{digest}"


def _default_retrieval_plan(
    requested_service: str,
    service_slug: str,
    chart_text: str,
) -> RetrievalPlan:
    note = chart_text.lower()
    has_red_flags = any(
        term in note for term in ("urinary", "retention", "saddle", "cauda", "bowel dysfunction", "bilateral leg weakness")
    )
    has_conservative_context = any(
        term in note for term in ("weeks", "physical therapy", "pt", "nsaids", "conservative", "persistent")
    )
    queries = [f"{requested_service} medical necessity policy criteria"]
    if has_red_flags:
        queries.extend(
            [
                f"{requested_service} cauda equina urinary retention saddle anesthesia",
                f"{requested_service} urgent neurologic deficit red flags imaging policy",
            ]
        )
    elif has_conservative_context:
        queries.extend(
            [
                f"{requested_service} conservative therapy duration failed physical therapy criteria",
                f"{requested_service} documentation gaps imaging management policy",
            ]
        )
    else:
        queries.extend(
            [
                f"{requested_service} imaging affects management documentation criteria",
                f"{requested_service} neurologic deficit policy criteria",
            ]
        )
    return RetrievalPlan(
        requested_service=requested_service,
        service_slug=service_slug,
        domain_filter="radiology_prior_auth",
        queries=queries,
    )


def _hybrid_score(query: str, document: str, metadata: dict, distance: float | None) -> float:
    vector_score = 1.0 / (1.0 + float(distance or 1.0))
    query_terms = _tokenize(query)
    doc_terms = _tokenize(document)
    keyword_overlap = len(query_terms & doc_terms)
    exact_bonus = 0.0
    if metadata.get("service") == "lumbar_mri" and "lumbar" in query.lower():
        exact_bonus += 0.5
    if "cauda equina" in document.lower() and any(term in query.lower() for term in ("cauda", "saddle", "retention")):
        exact_bonus += 0.75
    if "conservative therapy" in document.lower() and any(
        term in query.lower() for term in ("conservative", "physical therapy", "failed", "weeks")
    ):
        exact_bonus += 0.6
    if "documentation gaps" in document.lower() and "documentation" in query.lower():
        exact_bonus += 0.4
    return vector_score + (keyword_overlap * KEYWORD_WEIGHT) + exact_bonus


def _keyword_retrieve(plan: RetrievalPlan, *, top_k: int) -> list[RetrievedPolicyChunk]:
    chunks = _load_policy_chunks()
    scored: dict[str, tuple[dict, float]] = {}
    for query in plan.queries:
        query_terms = _tokenize(query)
        for chunk in chunks:
            if chunk["domain"] != plan.domain_filter:
                continue
            if plan.service_slug != "unknown" and chunk["service"] not in {plan.service_slug, "general"}:
                continue
            keyword_overlap = len(query_terms & _tokenize(chunk["text"]))
            if keyword_overlap == 0:
                continue
            score = (keyword_overlap * 0.25) + _hybrid_score(query, chunk["text"], chunk, 1.0)
            current = scored.get(chunk["chunk_id"])
            if current is None or score > current[1]:
                scored[chunk["chunk_id"]] = (chunk, score)

    ranked = [
        RetrievedPolicyChunk(
            chunk_id=chunk["chunk_id"],
            source_file=chunk["source_file"],
            section_title=chunk["section_title"],
            text=chunk["text"],
            score=round(score, 3),
            domain=chunk["domain"],
            service=chunk["service"],
            pathway=chunk["pathway"],
            heading_level=int(chunk["heading_level"]),
        )
        for chunk, score in scored.values()
    ]
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def _tokenize(text: str) -> set[str]:
    tokens = {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}
    return {token for token in tokens if token not in STOPWORDS}


def _can_use_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("DEMO_MODE", "false").lower() != "true"


def _get_query_client() -> OpenAI:
    global _QUERY_CLIENT
    if _QUERY_CLIENT is None:
        _QUERY_CLIENT = OpenAI()
    return _QUERY_CLIENT


def _embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_query_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]
