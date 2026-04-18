import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha1
from typing import Annotated
from urllib.parse import urlparse, urlunparse

from fastapi import APIRouter, Depends, HTTPException

from config.settings import Settings, get_settings
from schemas.requests import (
    CritiqueResponse,
    DiagramRequest,
    DiagramResponse,
    HealthResponse,
    PageContent,
    SourceConflict,
    SummarizeRequest,
    SummarizeResponse,
    infer_source_type,
)
from services.llm_service import LLMService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.reranker_service import RerankerService
from services.retrieval_cache import CachedCorpus, RetrievalCache
from services.vector_store import VectorStore
from services.token_counter import TokenCounter

router = APIRouter(prefix="/api", tags=["summarization"])
logger = logging.getLogger(__name__)

QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "show", "that", "the",
    "this", "to", "use", "what", "when", "where", "which", "who", "why", "with",
    "you", "your"
}

RETRIEVAL_MODE_ALIASES = {"answer", "critique", "diagram"}
SOURCE_TYPE_WEIGHTS = {
    "answer": {"confluence": 1.02, "jira": 0.99, "generic": 0.97},
    "critique": {"confluence": 1.08, "jira": 0.93, "generic": 1.01},
    "diagram": {"confluence": 1.06, "jira": 0.89, "generic": 1.02},
}
DOCUMENT_ROLE_HINTS = {
    "prd": ("prd", "product requirement", "requirements", "specification", "acceptance criteria"),
    "spec": ("spec", "technical spec", "tech spec", "functional spec", "contract"),
    "design": ("design", "architecture", "hld", "lld", "sequence", "flow", "diagram"),
    "api": ("api", "endpoint", "schema", "swagger", "graphql", "payload"),
    "qa": ("qa", "test plan", "test case", "uat", "validation", "acceptance test"),
    "ticket": ("jira", "story", "ticket", "issue", "bug", "task", "epic"),
    "runbook": ("runbook", "playbook", "operation", "ops", "deployment", "release"),
    "incident": ("incident", "rca", "postmortem", "outage", "sev", "root cause"),
}
DOCUMENT_ROLE_WEIGHTS = {
    "answer": {
        "prd": 1.07,
        "spec": 1.08,
        "design": 1.06,
        "api": 1.05,
        "qa": 0.99,
        "ticket": 1.0,
        "runbook": 0.97,
        "incident": 0.96,
        "generic": 1.0,
    },
    "critique": {
        "prd": 1.12,
        "spec": 1.11,
        "design": 1.07,
        "api": 1.06,
        "qa": 1.04,
        "ticket": 0.95,
        "runbook": 0.96,
        "incident": 0.98,
        "generic": 1.0,
    },
    "diagram": {
        "prd": 1.02,
        "spec": 1.08,
        "design": 1.15,
        "api": 1.08,
        "qa": 0.93,
        "ticket": 0.9,
        "runbook": 0.94,
        "incident": 0.9,
        "generic": 1.0,
    },
}
CONFLICT_TOPIC_HINTS = {
    "timeout": {
        "timeout", "ttl", "expire", "expires", "expiration", "refresh", "lifetime", "duration", "validity"
    },
    "rate_limit": {
        "rate", "limit", "throttle", "throttling", "qps", "rpm", "quota", "requests", "minute"
    },
    "retry_policy": {
        "retry", "retries", "backoff", "attempt", "attempts", "interval", "cooldown"
    },
    "tenancy": {
        "tenant", "tenancy", "deployment", "single-tenant", "multi-tenant", "single", "multi"
    },
    "status": {
        "enabled", "disabled", "required", "optional", "public", "private", "sync", "async",
        "synchronous", "asynchronous", "draft", "final", "deprecated", "supported", "unsupported"
    },
}
CONFLICT_VALUE_PATTERNS = [
    re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?%\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|secs|seconds|min|mins|minutes|hours|hrs|days)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:enabled|disabled|required|optional|public|private|sync|async|synchronous|asynchronous|draft|final|deprecated|supported|unsupported|single-tenant|multi-tenant|single tenant|multi tenant)\b",
        re.IGNORECASE,
    ),
]
CONFLICT_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class PageProfile:
    page: PageContent
    original_index: int
    role: str
    context_role: str
    priority: float
    title_overlap: float
    heading_overlap: float
    url_overlap: float
    source_weight: float
    role_weight: float
    canonical_url: str
    content_signature: str


def _shorten_text(value: str, limit: int = 160) -> str:
    normalized = re.sub(r"\s+", " ", (value or "").strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _format_profile_label(profile: PageProfile) -> str:
    return f"{_shorten_text(profile.page.title, 48)} [{profile.role}/{profile.page.source_type}]"


def _question_excerpt(user_question: str | None, limit: int = 120) -> str:
    return _shorten_text(user_question or "", limit) or "-"


def get_llm_service() -> LLMService:
    """Dependency injection for LLM service"""
    try:
        return LLMService()
    except Exception:
        logger.exception("Failed to create LLM service dependency.")
        raise


def get_chunking_service() -> ChunkingService:
    """Dependency injection for chunking service"""
    return ChunkingService(chunk_size=1000, chunk_overlap=200)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Dependency injection for embedding service"""
    settings = get_settings()
    try:
        return EmbeddingService(settings)
    except Exception:
        logger.exception(
            "Failed to create embedding service dependency provider=%s model=%s base_url=%s",
            settings.embedding_provider,
            settings.resolved_embedding_model(),
            settings.resolved_openai_base_url(),
        )
        raise


def get_vector_store() -> VectorStore:
    """Dependency injection for vector store"""
    settings = get_settings()
    return VectorStore(embedding_dimension=settings.resolved_embedding_dimensions())


@lru_cache
def get_reranker_service() -> RerankerService:
    """Dependency injection for optional cross-encoder reranker"""
    settings = get_settings()
    return RerankerService(settings)


def get_token_counter() -> TokenCounter:
    """Dependency injection for token counter"""
    settings = get_settings()
    return TokenCounter(settings)


@lru_cache
def get_retrieval_cache() -> RetrievalCache:
    """Dependency injection for in-memory retrieval cache"""
    return RetrievalCache()


def _tokenize_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*", text.lower())
    return [token for token in tokens if len(token) >= 3 and token not in QUERY_STOP_WORDS]


def _normalize_url_for_matching(url: str) -> str:
    raw_url = (url or "").strip()
    if not raw_url:
        return ""

    try:
        parsed = urlparse(raw_url)
        normalized_path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/")
        normalized_path = normalized_path.lower() or "/"
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                normalized_path,
                "",
                "",
                "",
            )
        )
    except Exception:
        return re.sub(r"/+$", "", raw_url.lower())


def _content_signature(markdown: str) -> str:
    normalized = re.sub(r"\s+", " ", markdown.lower()).strip()
    return sha1(normalized.encode("utf-8")).hexdigest()


def _extract_heading_terms(markdown: str) -> list[str]:
    headings = re.findall(r"^\s{0,3}#{1,6}\s+(.+)$", markdown, flags=re.MULTILINE)
    return _tokenize_text(" ".join(headings[:12]))


def _infer_retrieval_mode(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    return normalized if normalized in RETRIEVAL_MODE_ALIASES else "answer"


def _infer_document_role(page: PageContent) -> str:
    haystack = f"{page.title}\n{page.url}\n{page.markdown[:1200]}".lower()

    for role, hints in DOCUMENT_ROLE_HINTS.items():
        if any(hint in haystack for hint in hints):
            return role

    if page.source_type == "jira":
        return "ticket"

    return "generic"


def _infer_context_role(page: PageContent) -> str:
    if not isinstance(page.metadata, dict):
        return "default"

    raw_value = str(page.metadata.get("context_role") or "").strip().lower()
    if raw_value in {"current_page", "compare_anchor", "reference_context"}:
        return raw_value

    return "default"


def _infer_planner_mode(
    pages: list[PageContent],
    user_question: str,
    retrieval_mode: str
) -> tuple[str, str]:
    normalized_question = (user_question or "").strip().lower()
    context_roles = {_infer_context_role(page) for page in pages}

    if "compare_anchor" in context_roles:
        return (
            "compare-current-vs-pinned",
            "Kept the current page as the anchor, then balanced evidence from the other pinned references."
        )

    if "current_page" in context_roles:
        if retrieval_mode == "critique":
            return (
                "current-page-critique",
                "Focused the review on the active page instead of mixing in the wider context basket."
            )
        if retrieval_mode == "diagram":
            return (
                "current-page-diagram",
                "Built the diagram from the active page first so the visual stays anchored to the page in front of the user."
            )
        return (
            "current-page-summary",
            "Focused retrieval on the active page because the question pointed at the current tab."
        )

    if retrieval_mode == "diagram":
        return (
            "diagram-synthesis",
            "Favored architecture and flow-heavy sources to keep the diagram grounded and readable."
        )

    if retrieval_mode == "critique":
        return (
            "critique-review",
            "Favored requirement, spec, and validation-heavy sources to review for gaps and risks."
        )

    if re.search(r"\b(compare|difference|different|diff|versus|vs\.?)\b", normalized_question):
        return (
            "cross-doc-compare",
            "Treated the request as a comparison and balanced evidence across multiple documents."
        )

    if len(pages) <= 1:
        return (
            "single-doc-focus",
            "Focused on a single document, so the top chunks stay tightly anchored to that source."
        )

    return (
        "multi-doc-synthesis",
        "Balanced the context basket so one long document does not drown out the rest of the answer."
    )


def _build_page_profile(
    page: PageContent,
    original_index: int,
    user_question: str,
    retrieval_mode: str,
    planner_mode: str
) -> PageProfile:
    query_terms = set(_tokenize_text(user_question))
    title_terms = set(_tokenize_text(page.title))
    heading_terms = set(_extract_heading_terms(page.markdown))
    url_terms = set(_tokenize_text(_normalize_url_for_matching(page.url)))
    role = _infer_document_role(page)
    context_role = _infer_context_role(page)

    if query_terms:
        title_overlap = len(query_terms & title_terms) / len(query_terms)
        heading_overlap = len(query_terms & heading_terms) / len(query_terms)
        url_overlap = len(query_terms & url_terms) / len(query_terms)
    else:
        title_overlap = 0.0
        heading_overlap = 0.0
        url_overlap = 0.0

    source_weight = SOURCE_TYPE_WEIGHTS[retrieval_mode].get(page.source_type, 1.0)
    role_weight = DOCUMENT_ROLE_WEIGHTS[retrieval_mode].get(role, 1.0)

    priority = source_weight * role_weight
    priority += (title_overlap * 0.20) + (heading_overlap * 0.12) + (url_overlap * 0.06)

    if page.source_type == "jira" and re.search(r"\b[A-Z]{2,}-\d+\b", user_question):
        issue_key_match = re.search(r"\b([A-Z]{2,}-\d+)\b", user_question)
        if issue_key_match and issue_key_match.group(1).lower() in f"{page.title} {page.url}".lower():
            priority += 0.14

    if context_role == "compare_anchor":
        priority += 0.18
    elif context_role == "current_page":
        priority += 0.16
    elif context_role == "reference_context":
        priority += 0.03

    if planner_mode.startswith("compare") and original_index == 0:
        priority += 0.08
    elif planner_mode.startswith("current-page") and original_index == 0:
        priority += 0.06

    priority = max(0.80, min(1.35, priority))

    return PageProfile(
        page=page,
        original_index=original_index,
        role=role,
        context_role=context_role,
        priority=priority,
        title_overlap=title_overlap,
        heading_overlap=heading_overlap,
        url_overlap=url_overlap,
        source_weight=source_weight,
        role_weight=role_weight,
        canonical_url=_normalize_url_for_matching(page.url),
        content_signature=_content_signature(page.markdown),
    )


def _prepare_context_pages(
    pages: list[PageContent],
    user_question: str,
    retrieval_mode: str
) -> tuple[list[PageContent], dict[str, PageProfile], dict]:
    planner_mode, planner_note = _infer_planner_mode(pages, user_question, retrieval_mode)
    profiles = [
        _build_page_profile(page, index, user_question, retrieval_mode, planner_mode)
        for index, page in enumerate(pages)
    ]

    unique_by_url: dict[str, PageProfile] = {}
    unique_by_content: dict[str, PageProfile] = {}
    selected_profiles: list[PageProfile] = []
    ignored_labels: list[str] = []
    removed_by_url = 0
    removed_by_content = 0

    sorted_profiles = sorted(
        profiles,
        key=lambda profile: (
            profile.priority,
            len(profile.page.markdown),
            -profile.original_index,
        ),
        reverse=True,
    )

    for profile in sorted_profiles:
        if profile.canonical_url and profile.canonical_url in unique_by_url:
            removed_by_url += 1
            ignored_labels.append(f"{_format_profile_label(profile)} (duplicate URL)")
            continue

        if profile.content_signature in unique_by_content:
            removed_by_content += 1
            ignored_labels.append(f"{_format_profile_label(profile)} (duplicate content)")
            continue

        selected_profiles.append(profile)

        if profile.canonical_url:
            unique_by_url[profile.canonical_url] = profile
        unique_by_content[profile.content_signature] = profile

    selected_profiles.sort(key=lambda profile: profile.original_index)

    prepared_pages = [profile.page for profile in selected_profiles]
    page_profiles = {profile.page.url: profile for profile in selected_profiles}
    source_type_counts: dict[str, int] = {}

    for profile in selected_profiles:
        source_type_counts[profile.page.source_type] = source_type_counts.get(profile.page.source_type, 0) + 1

    stats = {
        "input_pages": len(pages),
        "prepared_pages": len(prepared_pages),
        "deduped_pages": removed_by_url + removed_by_content,
        "deduped_by_url": removed_by_url,
        "deduped_by_content": removed_by_content,
        "high_priority_pages": sum(1 for profile in selected_profiles if profile.priority >= 1.12),
        "retrieval_mode": retrieval_mode,
        "planner_mode": planner_mode,
        "planner_note": planner_note,
        "routing_note": (
            f"Prioritized {sum(1 for profile in selected_profiles if profile.priority >= 1.12)} "
            f"higher-signal pages using title/query match, document role, and source type."
        ),
        "source_diversity_summary": ", ".join(
            f"{source_type}:{count}" for source_type, count in sorted(source_type_counts.items())
        ),
        "ignored_pages_summary": "; ".join(ignored_labels[:3]) if ignored_labels else "",
    }

    return prepared_pages, page_profiles, stats


def _build_selection_summaries(
    retrieved_chunks: list,
    filtered_results: list[tuple],
    page_profiles: dict[str, PageProfile]
) -> dict[str, str]:
    selected_page_urls = {chunk.metadata.page_url for chunk in retrieved_chunks}
    selected_page_order: list[str] = []
    selected_labels: list[str] = []
    omitted_labels: list[str] = []

    for chunk in retrieved_chunks:
        if chunk.metadata.page_url not in selected_page_order:
            selected_page_order.append(chunk.metadata.page_url)

    for page_url in selected_page_order:
        profile = page_profiles.get(page_url)
        if profile:
            selected_labels.append(_format_profile_label(profile))

    omitted_candidates: list[tuple[float, PageProfile]] = []
    for chunk, score in filtered_results:
        profile = page_profiles.get(chunk.metadata.page_url)
        if not profile or chunk.metadata.page_url in selected_page_urls:
            continue
        omitted_candidates.append((score, profile))

    seen_omitted_urls = set()
    for _, profile in sorted(omitted_candidates, key=lambda item: item[0], reverse=True):
        if profile.page.url in seen_omitted_urls:
            continue
        seen_omitted_urls.add(profile.page.url)
        omitted_labels.append(f"{_format_profile_label(profile)} (lower-ranked)")
        if len(omitted_labels) >= 3:
            break

    return {
        "selected_sources_summary": "; ".join(selected_labels[:4]) if selected_labels else "",
        "omitted_sources_summary": "; ".join(omitted_labels) if omitted_labels else "",
    }


def _normalize_conflict_topic(topic: str) -> str:
    return topic.replace("_", " ").title()


def _extract_conflict_values(sentence: str) -> list[str]:
    values: list[str] = []

    for pattern in CONFLICT_VALUE_PATTERNS:
        for match in pattern.findall(sentence):
            normalized = re.sub(r"\s+", " ", str(match).strip().lower())
            if normalized and normalized not in values:
                values.append(normalized)

    return values


def _infer_conflict_topic(tokens: set[str]) -> str | None:
    for topic, hints in CONFLICT_TOPIC_HINTS.items():
        if tokens & hints:
            return topic
    return None


def _extract_conflict_candidates(retrieved_chunks: list) -> list[dict]:
    candidates: list[dict] = []

    for chunk in retrieved_chunks:
        sentences = CONFLICT_SENTENCE_SPLIT_RE.split(chunk.text)

        for sentence in sentences:
            normalized_sentence = re.sub(r"\s+", " ", sentence.strip())
            if len(normalized_sentence) < 30 or len(normalized_sentence) > 260:
                continue

            if normalized_sentence.startswith("#") or normalized_sentence.startswith("["):
                continue

            sentence_tokens = set(_tokenize_text(normalized_sentence))
            if len(sentence_tokens) < 3:
                continue

            topic = _infer_conflict_topic(sentence_tokens)
            if not topic:
                continue

            values = _extract_conflict_values(normalized_sentence)
            if not values:
                continue

            topic_tokens = CONFLICT_TOPIC_HINTS.get(topic, set())
            value_tokens = set(_tokenize_text(" ".join(values)))
            subject_tokens = [
                token for token in sentence_tokens
                if token not in topic_tokens and token not in value_tokens
            ]

            if len(subject_tokens) < 1:
                continue

            candidates.append(
                {
                    "topic": topic,
                    "page_title": chunk.metadata.page_title,
                    "page_url": chunk.metadata.page_url,
                    "sentence": normalized_sentence,
                    "values": values,
                    "value_key": "|".join(values),
                    "subject_tokens": set(subject_tokens),
                }
            )

    return candidates


def _detect_source_conflicts(retrieved_chunks: list) -> list[SourceConflict]:
    candidates = _extract_conflict_candidates(retrieved_chunks)
    conflicts: list[tuple[int, SourceConflict]] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            if left["page_url"] == right["page_url"]:
                continue

            if left["topic"] != right["topic"]:
                continue

            if left["value_key"] == right["value_key"]:
                continue

            subject_overlap = left["subject_tokens"] & right["subject_tokens"]
            if len(subject_overlap) < 1:
                continue

            # Avoid flagging noisy version mentions unless the subject is clearly shared.
            if left["topic"] == "status" and len(subject_overlap) < 2:
                continue

            pair_key = tuple(sorted((left["page_url"], right["page_url"]))) + (left["topic"],)
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            shared_subject = ", ".join(sorted(subject_overlap)[:3]) or _normalize_conflict_topic(left["topic"]).lower()
            summary = (
                f"{_normalize_conflict_topic(left['topic'])} looks inconsistent for {shared_subject}: "
                f"{left['page_title']} says {' / '.join(left['values'])}, "
                f"while {right['page_title']} says {' / '.join(right['values'])}."
            )

            conflicts.append(
                (
                    len(subject_overlap),
                    SourceConflict(
                        topic=_normalize_conflict_topic(left["topic"]),
                        summary=summary,
                        source_a_title=left["page_title"],
                        source_a_evidence=_shorten_text(left["sentence"], 180),
                        source_b_title=right["page_title"],
                        source_b_evidence=_shorten_text(right["sentence"], 180),
                    ),
                )
            )

    conflicts.sort(key=lambda item: item[0], reverse=True)
    return [conflict for _, conflict in conflicts[:3]]


def _build_chunk_embedding_text(chunk) -> str:
    return (
        f"Title: {chunk.metadata.page_title}\n"
        f"Source type: {chunk.metadata.source_type}\n"
        f"Content:\n{chunk.text}"
    )


def _fallback_page_url(page: PageContent, index: int) -> str:
    if page.url:
        return page.url

    slug = re.sub(r"[^a-z0-9]+", "-", page.title.lower()).strip("-")
    slug = slug or f"page-{index}"
    return f"local://{slug}-{index}"


def _prepare_pages(raw_pages: list[PageContent]) -> list[PageContent]:
    prepared_pages: list[PageContent] = []

    for index, page in enumerate(raw_pages, start=1):
        markdown = (page.markdown or "").strip()
        if not markdown:
            continue

        title = (page.title or "").strip() or f"Page {index}"
        url = (page.url or "").strip() or _fallback_page_url(page, index)
        source_type = (page.source_type or "").strip().lower() or infer_source_type(url)
        metadata = page.metadata if isinstance(page.metadata, dict) else {}

        prepared_pages.append(
            PageContent(
                title=title,
                url=url,
                markdown=markdown,
                source_type=source_type,
                metadata=metadata,
            )
        )

    return prepared_pages


def _calculate_hybrid_score(user_question: str, chunk, dense_score: float) -> float:
    query_terms = set(_tokenize_text(user_question))
    if not query_terms:
        return dense_score

    chunk_terms = set(_tokenize_text(chunk.text))
    title_terms = set(_tokenize_text(chunk.metadata.page_title))

    lexical_overlap = len(query_terms & chunk_terms) / len(query_terms)
    title_overlap = len(query_terms & title_terms) / len(query_terms)

    hybrid_score = (
        (dense_score * 0.72) +
        (lexical_overlap * 0.20) +
        (title_overlap * 0.08)
    )

    return hybrid_score


def _calculate_lexical_score(user_question: str, chunk, page_profiles: dict[str, PageProfile]) -> float:
    query_terms = set(_tokenize_text(user_question))
    if not query_terms:
        return 0.0

    chunk_terms = set(_tokenize_text(chunk.text))
    title_terms = set(_tokenize_text(chunk.metadata.page_title))
    profile = page_profiles.get(chunk.metadata.page_url)
    heading_overlap = profile.heading_overlap if profile else 0.0

    chunk_overlap = len(query_terms & chunk_terms) / len(query_terms)
    title_overlap = len(query_terms & title_terms) / len(query_terms)

    return (chunk_overlap * 0.70) + (title_overlap * 0.20) + (heading_overlap * 0.10)


def _lexical_search_chunks(
    user_question: str,
    chunks: list,
    page_profiles: dict[str, PageProfile],
    top_k: int
) -> list[tuple]:
    scored_results = []

    for chunk in chunks:
        score = _calculate_lexical_score(user_question, chunk, page_profiles)
        if score <= 0:
            continue
        scored_results.append((chunk, score))

    scored_results.sort(key=lambda item: item[1], reverse=True)
    return scored_results[:top_k]


def _priority_fallback_chunks(
    chunks: list,
    page_profiles: dict[str, PageProfile],
    top_k: int
) -> list[tuple]:
    chunk_lookup: dict[str, list] = {}

    for chunk in chunks:
        chunk_lookup.setdefault(chunk.metadata.page_url, []).append(chunk)

    ranked_profiles = sorted(
        page_profiles.values(),
        key=lambda profile: (profile.priority, profile.title_overlap, profile.heading_overlap),
        reverse=True,
    )

    fallback_results: list[tuple] = []
    seen_chunk_keys = set()

    for profile in ranked_profiles:
        page_chunks = chunk_lookup.get(profile.page.url, [])
        for chunk in page_chunks[:2]:
            chunk_key = _chunk_key(chunk)
            if chunk_key in seen_chunk_keys:
                continue
            seen_chunk_keys.add(chunk_key)
            fallback_results.append((chunk, max(0.12, profile.priority * 0.1)))
            if len(fallback_results) >= top_k:
                return fallback_results

    return fallback_results


def _is_forbidden_provider_error(exc: str | Exception) -> bool:
    message = str(exc).lower()
    return (
        "selected provider is forbidden" in message or
        "forbidden" in message or
        "403" in message
    )


def _provider_error_detail(
    *,
    stage: str,
    model_name: str,
    settings: Settings,
    error: Exception
) -> str:
    gateway = settings.resolved_openai_base_url() if settings.resolved_cloud_provider() == "openai" else None
    gateway_hint = f" via gateway {gateway}" if gateway else ""

    return (
        f"Gateway forbids the {stage} model '{model_name}' for provider "
        f"'{settings.resolved_cloud_provider()}'{gateway_hint}. Original error: {error}"
    )


def _raise_provider_http_error_if_needed(
    *,
    stage: str,
    model_name: str,
    settings: Settings,
    error: Exception
) -> None:
    if _is_forbidden_provider_error(error):
        raise HTTPException(
            status_code=502,
            detail=_provider_error_detail(
                stage=stage,
                model_name=model_name,
                settings=settings,
                error=error
            )
        )


def _apply_page_quality_scores(
    scored_chunks: list[tuple],
    page_profiles: dict[str, PageProfile]
) -> list[tuple]:
    reranked: list[tuple] = []

    for chunk, base_score in scored_chunks:
        profile = page_profiles.get(chunk.metadata.page_url)
        if profile is None:
            reranked.append((chunk, base_score))
            continue

        adjusted_score = base_score * profile.priority
        adjusted_score += (profile.title_overlap * 0.08) + (profile.heading_overlap * 0.05)

        if chunk.metadata.chunk_index == 0 and profile.role in {"prd", "spec", "design", "api"}:
            adjusted_score += 0.02

        reranked.append((chunk, adjusted_score))

    reranked.sort(
        key=lambda item: (
            item[1],
            page_profiles.get(item[0].metadata.page_url).priority
            if item[0].metadata.page_url in page_profiles
            else 1.0,
        ),
        reverse=True,
    )
    return reranked


def _build_page_chunk_caps(page_profiles: dict[str, PageProfile]) -> dict[str, int]:
    caps: dict[str, int] = {}

    for page_url, profile in page_profiles.items():
        if profile.priority >= 1.18:
            caps[page_url] = 4
        elif profile.priority >= 1.04:
            caps[page_url] = 3
        else:
            caps[page_url] = 2

    return caps


def _build_source_type_chunk_caps(
    page_profiles: dict[str, PageProfile],
    planner_mode: str
) -> dict[str, int]:
    source_page_counts: dict[str, int] = {}

    for profile in page_profiles.values():
        source_page_counts[profile.page.source_type] = source_page_counts.get(profile.page.source_type, 0) + 1

    if len(source_page_counts) <= 1:
        return {}

    caps: dict[str, int] = {}
    compare_mode = planner_mode.startswith("compare")
    current_focus_mode = planner_mode.startswith("current-page")

    for source_type, page_count in source_page_counts.items():
        if compare_mode:
            cap = 2 if page_count <= 2 else 3
        elif current_focus_mode:
            cap = 2
        else:
            cap = 3 if page_count <= 3 else 4

        caps[source_type] = cap

    return caps


def _build_context_budget_policy(
    settings: Settings,
    retrieval_mode: str,
    planner_mode: str
) -> dict[str, object]:
    model_route = settings.resolved_model_route(retrieval_mode)
    target_model = settings.resolved_model_for_task(retrieval_mode)
    compare_mode = planner_mode.startswith("compare")
    single_page_focus_mode = planner_mode.startswith("current-page") or planner_mode.startswith("single-doc")

    policy: dict[str, object] = {
        "policy_name": "default-shared-context",
        "provider": settings.llm_provider,
        "model_route": model_route,
        "target_model": target_model,
        "max_candidates": 20,
        "max_total_chunks": None,
        "max_chunks_per_page": 3,
        "max_input_tokens_override": None,
        "max_stitched_chunks": None,
        "stitch_edge_radius": None,
        "stitch_max_bridge_gap": None,
    }

    if settings.llm_provider == "ollama":
        if model_route == "summary":
            policy.update(
                {
                    "policy_name": "ollama-summary-tight",
                    "max_candidates": 12,
                    "max_total_chunks": 5,
                    "max_chunks_per_page": 2,
                    "max_input_tokens_override": 5200,
                    "max_stitched_chunks": 2,
                    "stitch_edge_radius": 1,
                    "stitch_max_bridge_gap": 1,
                }
            )
        else:
            policy.update(
                {
                    "policy_name": "ollama-reasoning-balanced",
                    "max_candidates": 14,
                    "max_total_chunks": 7 if not compare_mode else 6,
                    "max_chunks_per_page": 3,
                    "max_input_tokens_override": 7600 if not compare_mode else 6800,
                    "max_stitched_chunks": 3,
                    "stitch_edge_radius": 1,
                    "stitch_max_bridge_gap": 2 if single_page_focus_mode else 1,
                }
            )
    elif model_route == "reasoning" and settings.resolved_reasoning_model() != settings.resolved_summary_model():
        policy.update(
            {
                "policy_name": "reasoning-route-balanced",
                "max_candidates": 16,
                "max_total_chunks": 8 if not compare_mode else 7,
                "max_chunks_per_page": 3,
                "max_input_tokens_override": 12000 if not compare_mode else 10000,
                "max_stitched_chunks": 3,
            }
        )

    return policy


def _build_low_confidence_next_step(
    user_question: str | None,
    retrieval_mode: str,
    retrieval_stats: dict
) -> str:
    query_terms = _tokenize_text(user_question or "")[:3]
    query_hint = f" that explicitly mentions {', '.join(query_terms)}" if query_terms else ""

    if retrieval_stats.get("input_pages", 0) >= 3 and retrieval_stats.get("selected_pages", 0) <= 1:
        return "Remove unrelated pages or ask to focus on the current page so the context basket is tighter."

    if retrieval_mode == "critique":
        return (
            "Pin acceptance criteria, QA notes, or edge-case details"
            f"{query_hint} before retrying."
        )

    if retrieval_mode == "diagram":
        return (
            "Pin an architecture, API, or flow document"
            f"{query_hint} before asking for a diagram again."
        )

    return (
        "Pin a more direct PRD, spec, or source page"
        f"{query_hint}, or rephrase the question using the document's own terms."
    )


def _build_confidence_assessment(
    user_question: str | None,
    retrieval_mode: str,
    retrieval_stats: dict
) -> dict[str, str | bool]:
    hard_reasons: list[str] = []
    supporting_reasons: list[str] = []

    avg_similarity = float(retrieval_stats.get("avg_similarity") or 0)
    candidates_found = int(retrieval_stats.get("candidates_found") or 0)
    input_pages = int(retrieval_stats.get("input_pages") or 0)
    selected_pages = int(retrieval_stats.get("selected_pages") or 0)

    if retrieval_stats.get("retrieval_no_match_fallback_used"):
        hard_reasons.append(
            "The question had weak overlap with the pinned pages, so retrieval fell back to top-ranked context."
        )

    if retrieval_stats.get("similarity_fallback_used"):
        hard_reasons.append(
            "Similarity scores were weak, so the assistant used the best available excerpts instead of strong matches."
        )
    elif avg_similarity and avg_similarity < 0.28:
        hard_reasons.append(
            "The retrieved excerpts only weakly matched the question."
        )

    if retrieval_stats.get("retrieval_strategy") == "lexical-fallback":
        supporting_reasons.append(
            "Retrieval relied on keyword overlap because embeddings were unavailable."
        )

    if input_pages >= 3 and selected_pages <= 1:
        supporting_reasons.append(
            "Most evidence came from a single page inside a larger context basket."
        )

    if candidates_found <= 3:
        supporting_reasons.append(
            "Only a small number of relevant chunks were found."
        )

    low_confidence = bool(hard_reasons) or len(supporting_reasons) >= 2

    if not low_confidence:
        return {
            "low_confidence": False,
            "confidence_level": "normal",
        }

    reason_parts = hard_reasons[:2] if hard_reasons else supporting_reasons[:2]
    confidence_note = "Answer may be incomplete. " + " ".join(reason_parts)

    return {
        "low_confidence": True,
        "confidence_level": "low",
        "confidence_note": confidence_note,
        "confidence_next_step": _build_low_confidence_next_step(
            user_question,
            retrieval_mode,
            retrieval_stats,
        ),
    }


def _rerank_candidates(user_question: str, candidates: list[tuple]) -> list[tuple]:
    reranked = []

    for chunk, dense_score in candidates:
        hybrid_score = _calculate_hybrid_score(user_question, chunk, dense_score)
        reranked.append((chunk, hybrid_score))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked


def _chunk_key(chunk) -> tuple[str, int]:
    return (chunk.metadata.page_url, chunk.metadata.chunk_index)


def _stitch_selected_chunks(
    selected_chunks: list,
    scored_chunks: list[tuple],
    all_chunks: list,
    page_profiles: dict[str, PageProfile],
    token_counter: TokenCounter,
    reserve_tokens: int,
    planner_mode: str,
    max_input_tokens_override: int | None = None,
    max_stitched_chunks: int | None = None,
    stitch_edge_radius: int | None = None,
    stitch_max_bridge_gap: int | None = None,
) -> tuple[list, dict[str, int | bool | str]]:
    if not selected_chunks:
        return selected_chunks, {
            "stitched_chunks_added": 0,
            "stitched_bridge_chunks_added": 0,
            "stitching_applied": False,
            "stitching_mode": "none",
        }

    max_input = (
        max_input_tokens_override
        if max_input_tokens_override is not None
        else token_counter.settings.max_input_tokens - reserve_tokens
    )
    total_tokens = token_counter.count_chunks(selected_chunks)
    selected_keys = {_chunk_key(chunk) for chunk in selected_chunks}
    chunk_lookup = {_chunk_key(chunk): chunk for chunk in all_chunks}
    score_lookup = {_chunk_key(chunk): score for chunk, score in scored_chunks}
    page_order: list[str] = []
    selected_indices_by_page: dict[str, set[int]] = {}

    for chunk in selected_chunks:
        page_url = chunk.metadata.page_url
        if page_url not in selected_indices_by_page:
            page_order.append(page_url)
            selected_indices_by_page[page_url] = set()
        selected_indices_by_page[page_url].add(chunk.metadata.chunk_index)

    compare_mode = planner_mode.startswith("compare")
    single_page_focus_mode = (
        planner_mode.startswith("current-page")
        or planner_mode.startswith("single-doc")
    )
    max_bridge_gap = stitch_max_bridge_gap if stitch_max_bridge_gap is not None else (2 if single_page_focus_mode else 1)
    edge_radius = stitch_edge_radius if stitch_edge_radius is not None else (1 if compare_mode else 2 if single_page_focus_mode else 1)
    stitch_mode = "bridge+adjacent" if max_bridge_gap > 0 else "adjacent"

    candidate_entries: list[tuple[tuple, object, str]] = []
    candidate_keys: set[tuple[str, int]] = set()

    def queue_candidate(
        candidate_chunk,
        strategy: str,
        distance: int,
        anchor_score: float,
        page_priority: float,
        page_rank: int,
    ) -> None:
        candidate_key = _chunk_key(candidate_chunk)
        if candidate_key in selected_keys or candidate_key in candidate_keys:
            return

        candidate_keys.add(candidate_key)
        strategy_rank = 0 if strategy == "bridge" else 1
        candidate_entries.append(
            (
                (
                    strategy_rank,
                    page_rank,
                    -page_priority,
                    -anchor_score,
                    distance,
                    candidate_chunk.metadata.chunk_index,
                ),
                candidate_chunk,
                strategy,
            )
        )

    for page_rank, page_url in enumerate(page_order):
        selected_indices = sorted(selected_indices_by_page.get(page_url, set()))
        if not selected_indices:
            continue

        page_priority = page_profiles.get(page_url).priority if page_url in page_profiles else 1.0

        for previous_index, next_index in zip(selected_indices, selected_indices[1:]):
            gap = next_index - previous_index - 1
            if gap <= 0 or gap > max_bridge_gap:
                continue

            left_score = score_lookup.get((page_url, previous_index), 0.0)
            right_score = score_lookup.get((page_url, next_index), 0.0)
            anchor_score = max(left_score, right_score)

            for middle_index in range(previous_index + 1, next_index):
                middle_key = (page_url, middle_index)
                middle_chunk = chunk_lookup.get(middle_key)
                if not middle_chunk:
                    continue
                queue_candidate(
                    middle_chunk,
                    strategy="bridge",
                    distance=gap,
                    anchor_score=anchor_score,
                    page_priority=page_priority,
                    page_rank=page_rank,
                )

        for selected_index in selected_indices:
            anchor_score = score_lookup.get((page_url, selected_index), 0.0)
            for offset in range(1, edge_radius + 1):
                for direction in (-1, 1):
                    neighbor_key = (page_url, selected_index + (direction * offset))
                    neighbor_chunk = chunk_lookup.get(neighbor_key)
                    if not neighbor_chunk:
                        continue
                    queue_candidate(
                        neighbor_chunk,
                        strategy="adjacent",
                        distance=offset,
                        anchor_score=anchor_score,
                        page_priority=page_priority,
                        page_rank=page_rank,
                    )

    added_chunks: dict[tuple[str, int], object] = {}
    stitched_chunks_added = 0
    bridge_chunks_added = 0

    for _, candidate_chunk, strategy in sorted(candidate_entries, key=lambda item: item[0]):
        candidate_key = _chunk_key(candidate_chunk)
        if candidate_key in selected_keys or candidate_key in added_chunks:
            continue

        if max_stitched_chunks is not None and stitched_chunks_added >= max_stitched_chunks:
            break

        candidate_tokens = token_counter.count_text(candidate_chunk.text)
        if total_tokens + candidate_tokens > max_input:
            continue

        added_chunks[candidate_key] = candidate_chunk
        total_tokens += candidate_tokens
        stitched_chunks_added += 1
        if strategy == "bridge":
            bridge_chunks_added += 1

    if not added_chunks:
        return selected_chunks, {
            "stitched_chunks_added": 0,
            "stitched_bridge_chunks_added": 0,
            "stitching_applied": False,
            "stitching_mode": stitch_mode,
        }

    ordered_chunks: list = []
    seen_keys: set[tuple[str, int]] = set()

    for page_url in page_order:
        page_chunk_keys = [
            chunk_key
            for chunk_key in sorted(selected_keys | set(added_chunks.keys()), key=lambda item: item[1])
            if chunk_key[0] == page_url
        ]
        for page_chunk_key in page_chunk_keys:
            if page_chunk_key in seen_keys:
                continue
            if page_chunk_key in added_chunks:
                ordered_chunks.append(added_chunks[page_chunk_key])
            else:
                ordered_chunks.append(chunk_lookup[page_chunk_key])
            seen_keys.add(page_chunk_key)

    for chunk in selected_chunks:
        current_key = _chunk_key(chunk)
        if current_key not in seen_keys:
            ordered_chunks.append(chunk)
            seen_keys.add(current_key)

    return ordered_chunks, {
        "stitched_chunks_added": stitched_chunks_added,
        "stitched_bridge_chunks_added": bridge_chunks_added,
        "stitching_applied": stitched_chunks_added > 0,
        "stitching_mode": stitch_mode,
    }


async def retrieve_relevant_chunks(
    request: SummarizeRequest,
    chunking_service: ChunkingService,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    reranker_service: RerankerService,
    retrieval_cache: RetrievalCache,
    token_counter: TokenCounter,
    reserve_tokens: int = 6000,
    retrieval_mode: str = "answer"
):
    """Run the common RAG retrieval pipeline and return selected chunks plus metadata"""
    retrieval_mode = _infer_retrieval_mode(retrieval_mode)
    prepared_pages, page_profiles, context_page_stats = _prepare_context_pages(
        request.pages,
        request.user_question,
        retrieval_mode
    )

    if not prepared_pages:
        raise HTTPException(
            status_code=400,
            detail="No content to process - all pages are empty"
        )

    settings = get_settings()
    planner_mode = context_page_stats.get("planner_mode", "multi-doc-synthesis")
    budget_policy = _build_context_budget_policy(
        settings=settings,
        retrieval_mode=retrieval_mode,
        planner_mode=planner_mode,
    )

    corpus_signature = retrieval_cache.build_corpus_signature(prepared_pages)
    cached_corpus = retrieval_cache.get_corpus(corpus_signature)
    corpus_cache_hit = cached_corpus is not None
    embeddings_available = False
    embedding_fallback_used = False
    embedding_fallback_reason = ""

    if cached_corpus is not None:
        chunks = cached_corpus.chunks
        chunk_stats = cached_corpus.chunk_stats
        embeddings = cached_corpus.embeddings
        embeddings_available = len(embeddings) == len(chunks) and len(embeddings) > 0
    else:
        chunks = chunking_service.chunk_pages(prepared_pages)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content to process - all pages are empty"
            )

        chunk_stats = chunking_service.get_chunk_stats(chunks)
        chunk_texts = [_build_chunk_embedding_text(chunk) for chunk in chunks]
        embeddings = []

        try:
            embeddings = await embedding_service.embed_texts(chunk_texts)
            embeddings_available = len(embeddings) == len(chunks)
        except Exception as exc:
            embedding_fallback_used = True
            embedding_fallback_reason = str(exc)
            embeddings = []
            logger.warning(
                "Chunk embedding failed; falling back to lexical retrieval mode=%s reason=%s",
                retrieval_mode,
                _shorten_text(str(exc), 220),
            )

        retrieval_cache.set_corpus(
            corpus_signature,
            CachedCorpus(
                chunks=chunks,
                embeddings=embeddings,
                chunk_stats=chunk_stats
            )
        )

    question_signature = retrieval_cache.build_query_signature(request.user_question)
    top_k_candidates = min(int(budget_policy["max_candidates"]), len(chunks))
    query_embedding = None
    query_embedding_cache_hit = False
    no_match_fallback_used = False

    if embeddings_available:
        vector_store.add_chunks(chunks, embeddings)
        question_embedding = retrieval_cache.get_query_embedding(question_signature)
        query_embedding_cache_hit = question_embedding is not None

        if question_embedding is None:
            try:
                question_embedding = await embedding_service.embed_single(request.user_question)
                retrieval_cache.set_query_embedding(question_signature, question_embedding)
            except Exception as exc:
                embedding_fallback_used = True
                embedding_fallback_reason = str(exc)
                question_embedding = None
                logger.warning(
                    "Question embedding failed; falling back to lexical retrieval mode=%s reason=%s",
                    retrieval_mode,
                    _shorten_text(str(exc), 220),
                )

    if embeddings_available and question_embedding is not None:
        candidate_results = vector_store.search(question_embedding, top_k=top_k_candidates)
    else:
        candidate_results = _lexical_search_chunks(
            request.user_question,
            chunks,
            page_profiles,
            top_k_candidates
        )

    if not candidate_results:
        no_match_fallback_used = True
        logger.info(
            "No retrieval candidates matched question; using priority fallback mode=%s question=%s",
            retrieval_mode,
            _question_excerpt(request.user_question),
        )
        candidate_results = _priority_fallback_chunks(
            chunks,
            page_profiles,
            top_k_candidates
        )

    if not candidate_results:
        raise HTTPException(
            status_code=400,
            detail="Pinned pages do not contain enough overlapping context for this question"
        )

    reranked_results = _rerank_candidates(request.user_question, candidate_results)
    reranked_results, reranker_stats = reranker_service.rerank(
        request.user_question,
        reranked_results
    )
    reranked_results = _apply_page_quality_scores(reranked_results, page_profiles)

    if not reranked_results:
        no_match_fallback_used = True
        logger.info(
            "Reranker removed all candidates; using priority fallback mode=%s question=%s",
            retrieval_mode,
            _question_excerpt(request.user_question),
        )
        reranked_results = _priority_fallback_chunks(
            chunks,
            page_profiles,
            top_k_candidates
        )

    min_similarity = 0.22
    filtered_results = [
        (chunk, score) for chunk, score in reranked_results
        if score > min_similarity
    ]
    similarity_fallback_used = False

    if not filtered_results:
        fallback_count = min(5, len(reranked_results))
        filtered_results = reranked_results[:fallback_count]
        similarity_fallback_used = bool(filtered_results)
        if similarity_fallback_used:
            logger.info(
                "Similarity threshold removed all chunks; using top reranked chunks mode=%s threshold=%.2f",
                retrieval_mode,
                min_similarity,
            )

    if not filtered_results:
        no_match_fallback_used = True
        logger.info(
            "Similarity fallback still empty; using priority fallback mode=%s question=%s",
            retrieval_mode,
            _question_excerpt(request.user_question),
        )
        filtered_results = _priority_fallback_chunks(
            chunks,
            page_profiles,
            top_k_candidates
        )

    if not filtered_results:
        raise HTTPException(
            status_code=400,
            detail="Pinned pages do not contain enough usable context for this question"
        )

    page_chunk_caps = _build_page_chunk_caps(page_profiles)
    max_chunks_per_page = int(budget_policy["max_chunks_per_page"])
    page_chunk_caps = {
        page_url: min(cap, max_chunks_per_page)
        for page_url, cap in page_chunk_caps.items()
    }
    source_type_chunk_caps = _build_source_type_chunk_caps(
        page_profiles,
        planner_mode
    )
    effective_context_budget_tokens = (
        int(budget_policy["max_input_tokens_override"])
        if budget_policy.get("max_input_tokens_override") is not None
        else token_counter.settings.max_input_tokens - reserve_tokens
    )
    retrieved_chunks = token_counter.select_chunks_within_budget(
        chunks_with_scores=filtered_results,
        reserve_tokens=reserve_tokens,
        max_chunks_per_page=max_chunks_per_page,
        page_chunk_caps=page_chunk_caps,
        source_type_chunk_caps=source_type_chunk_caps,
        max_total_chunks=budget_policy.get("max_total_chunks"),
        max_input_tokens_override=budget_policy.get("max_input_tokens_override"),
    )

    if not retrieved_chunks:
        raise HTTPException(
            status_code=400,
            detail="No chunks fit within token budget"
        )

    retrieved_chunks, stitching_stats = _stitch_selected_chunks(
        selected_chunks=retrieved_chunks,
        scored_chunks=filtered_results,
        all_chunks=chunks,
        page_profiles=page_profiles,
        token_counter=token_counter,
        reserve_tokens=reserve_tokens,
        planner_mode=planner_mode,
        max_input_tokens_override=budget_policy.get("max_input_tokens_override"),
        max_stitched_chunks=budget_policy.get("max_stitched_chunks"),
        stitch_edge_radius=budget_policy.get("stitch_edge_radius"),
        stitch_max_bridge_gap=budget_policy.get("stitch_max_bridge_gap"),
    )

    selected_tokens = token_counter.count_chunks(retrieved_chunks)
    selected_page_urls = {chunk.metadata.page_url for chunk in retrieved_chunks}
    source_conflicts = _detect_source_conflicts(retrieved_chunks)
    similarity_scores = [
        score for chunk, score in filtered_results
        if chunk in retrieved_chunks
    ]
    selection_summaries = _build_selection_summaries(
        retrieved_chunks=retrieved_chunks,
        filtered_results=filtered_results,
        page_profiles=page_profiles
    )

    retrieval_stats = {
        "chunk_stats": chunk_stats,
        "retrieved_chunks": len(retrieved_chunks),
        "candidates_found": len(candidate_results),
        "filtered_by_similarity": len(filtered_results),
        "avg_similarity": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
        "selected_chunk_tokens": selected_tokens,
        "selected_pages": len(selected_page_urls),
        "adjacent_chunks_added": stitching_stats["stitched_chunks_added"],
        "stitched_chunks_added": stitching_stats["stitched_chunks_added"],
        "stitched_bridge_chunks_added": stitching_stats["stitched_bridge_chunks_added"],
        "stitching_applied": stitching_stats["stitching_applied"],
        "stitching_mode": stitching_stats["stitching_mode"],
        "corpus_cache_hit": corpus_cache_hit,
        "query_embedding_cache_hit": query_embedding_cache_hit,
        "similarity_threshold": min_similarity,
        "similarity_fallback_used": similarity_fallback_used,
        "context_budget_tokens": effective_context_budget_tokens,
        "context_budget_policy": str(budget_policy["policy_name"]),
        "context_budget_chunk_cap": budget_policy.get("max_total_chunks"),
        "context_budget_candidate_cap": budget_policy.get("max_candidates"),
        "context_budget_page_cap": budget_policy.get("max_chunks_per_page"),
        "context_budget_model_route": str(budget_policy["model_route"]),
        "context_budget_target_model": str(budget_policy["target_model"]),
        "retrieval_strategy": "dense+hybrid" if embeddings_available and question_embedding is not None else "lexical-fallback",
        "embedding_fallback_used": embedding_fallback_used,
        "embedding_provider_forbidden": _is_forbidden_provider_error(embedding_fallback_reason) if embedding_fallback_reason else False,
        "retrieval_no_match_fallback_used": no_match_fallback_used,
        "source_type_diversity_caps_applied": bool(source_type_chunk_caps),
        "source_conflict_count": len(source_conflicts),
        "source_conflict_summary": source_conflicts[0].summary if source_conflicts else "",
    }
    retrieval_stats.update(context_page_stats)
    retrieval_stats.update(selection_summaries)
    retrieval_stats.update(reranker_stats)
    retrieval_stats.update(
        _build_confidence_assessment(
            request.user_question,
            retrieval_mode,
            retrieval_stats,
        )
    )
    logger.info(
        "Retrieval completed mode=%s input_pages=%s prepared_pages=%s chunks=%s retrieved_chunks=%s strategy=%s deduped_pages=%s corpus_cache_hit=%s query_embedding_cache_hit=%s embedding_fallback_used=%s no_match_fallback_used=%s",
        retrieval_mode,
        context_page_stats["input_pages"],
        context_page_stats["prepared_pages"],
        chunk_stats.get("total_chunks", len(chunks)),
        len(retrieved_chunks),
        retrieval_stats["retrieval_strategy"],
        context_page_stats["deduped_pages"],
        corpus_cache_hit,
        query_embedding_cache_hit,
        embedding_fallback_used,
        no_match_fallback_used,
    )

    return retrieved_chunks, retrieval_stats, source_conflicts


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_pages(
    request: SummarizeRequest, llm_service: Annotated[LLMService, Depends(get_llm_service)]
):
    """
    Summarize multiple web pages using AI

    - **pages**: List of pages with markdown content (1-10 pages)
    - **user_question**: Optional specific question to answer about the pages

    Returns summary with citations and token usage statistics.
    """
    try:
        settings = get_settings()
        prepared_pages = _prepare_pages(request.pages)
        logger.info(
            "Handling /api/summarize pages=%s question=%s",
            len(prepared_pages),
            _question_excerpt(request.user_question),
        )
        if not prepared_pages:
            raise HTTPException(
                status_code=400,
                detail="At least one page with markdown content is required"
            )

        user_question = (request.user_question or "").strip() or None

        # Validate total token count
        total_chars = sum(len(page.markdown) for page in prepared_pages)
        estimated_tokens = total_chars // 4  # Rough estimation

        if estimated_tokens > settings.max_input_tokens:
            raise HTTPException(
                status_code=400,
                detail=f"Input too large: approximately {estimated_tokens:,} tokens "
                f"(maximum: {settings.max_input_tokens:,} tokens)",
            )

        # Generate summary
        summary, citations, token_usage = await llm_service.summarize(
            pages=prepared_pages,
            user_question=user_question
        )

        return SummarizeResponse(
            summary=summary,
            citations=citations,
            token_usage=token_usage,
            model_used=llm_service.get_model_name(),
        )

    except HTTPException as exc:
        log_method = logger.error if exc.status_code >= 500 else logger.warning
        log_method("Request failed path=/api/summarize status=%s detail=%s", exc.status_code, exc.detail)
        raise
    except ValueError as e:
        logger.warning("Request validation failed path=/api/summarize detail=%s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(
            "Unexpected summarize error model=%s pages=%s question=%s",
            llm_service.get_model_name(),
            len(prepared_pages),
            _question_excerpt(request.user_question),
        )
        _raise_provider_http_error_if_needed(
            stage="chat",
            model_name=llm_service.get_model_name(),
            settings=settings,
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/rag-summarize", response_model=SummarizeResponse)
async def rag_summarize_pages(
    request: SummarizeRequest,
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    chunking_service: Annotated[ChunkingService, Depends(get_chunking_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    reranker_service: Annotated[RerankerService, Depends(get_reranker_service)],
    retrieval_cache: Annotated[RetrievalCache, Depends(get_retrieval_cache)],
    token_counter: Annotated[TokenCounter, Depends(get_token_counter)],
):
    """
    Phase 2: RAG-based summarization with chunking, embedding, and retrieval

    - **pages**: List of pages with markdown content
    - **user_question**: Required question to answer

    Flow:
    1. Chunk pages into smaller segments
    2. Generate embeddings for chunks
    3. Store in vector database
    4. Search for relevant chunks based on question
    5. Generate answer using only relevant chunks

    Returns summary with chunk-level citations and token usage.
    """
    try:
        request_started_at = time.perf_counter()
        settings = get_settings()
        prepared_pages = _prepare_pages(request.pages)
        logger.info(
            "Handling /api/rag-summarize pages=%s question=%s",
            len(prepared_pages),
            _question_excerpt(request.user_question),
        )
        if not prepared_pages:
            raise HTTPException(
                status_code=400,
                detail="At least one page with markdown content is required"
            )

        user_question = (request.user_question or "").strip()

        # Validate user question is provided
        if not user_question:
            raise HTTPException(
                status_code=400,
                detail="user_question is required for RAG-based summarization"
            )

        retrieval_started_at = time.perf_counter()
        retrieved_chunks, retrieval_stats, source_conflicts = await retrieve_relevant_chunks(
            request=SummarizeRequest(
                pages=prepared_pages,
                user_question=user_question
            ),
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            vector_store=vector_store,
            reranker_service=reranker_service,
            retrieval_cache=retrieval_cache,
            token_counter=token_counter,
            reserve_tokens=6000,
            retrieval_mode="answer"
        )
        retrieval_latency_ms = round((time.perf_counter() - retrieval_started_at) * 1000)

        llm_chunks = retrieved_chunks
        hybrid_prefilter_stats: dict[str, object] = {"hybrid_prefilter_applied": False}
        if settings.hybrid_mode_enabled():
            llm_chunks, hybrid_prefilter_stats = await llm_service.prefilter_chunks(
                retrieved_chunks,
                user_question,
                task_kind="answer",
                planner_mode=str(retrieval_stats.get("planner_mode") or ""),
            )
            hybrid_prefilter_stats["hybrid_prefilter_context_tokens"] = token_counter.count_chunks(llm_chunks)

        # Step 7: Generate summary using only retrieved chunks
        generation_started_at = time.perf_counter()
        summary, citations, token_usage, suggestions = await llm_service.summarize_from_chunks(
            chunks=llm_chunks,
            user_question=user_question
        )
        generation_latency_ms = round((time.perf_counter() - generation_started_at) * 1000)

        # Add chunking metadata to token usage
        token_usage.update(retrieval_stats)
        token_usage.update(hybrid_prefilter_stats)
        token_usage["retrieval_latency_ms"] = retrieval_latency_ms
        token_usage.setdefault("generation_latency_ms", generation_latency_ms)
        token_usage["end_to_end_latency_ms"] = round((time.perf_counter() - request_started_at) * 1000)

        # Estimate API cost
        cost_estimate = token_counter.estimate_cost(
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0)
        )
        token_usage["cost_estimate"] = cost_estimate

        return SummarizeResponse(
            summary=summary,
            citations=citations,
            source_conflicts=source_conflicts or None,
            token_usage=token_usage,
            model_used=llm_service.get_model_name(),
            suggestions=suggestions
        )

    except HTTPException as exc:
        log_method = logger.error if exc.status_code >= 500 else logger.warning
        log_method("Request failed path=/api/rag-summarize status=%s detail=%s", exc.status_code, exc.detail)
        raise
    except ValueError as e:
        logger.warning("Request validation failed path=/api/rag-summarize detail=%s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(
            "Unexpected RAG summarize error model=%s pages=%s question=%s",
            llm_service.get_model_name(),
            len(prepared_pages),
            _question_excerpt(request.user_question),
        )
        _raise_provider_http_error_if_needed(
            stage="chat",
            model_name=llm_service.get_model_name(),
            settings=settings,
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/critique", response_model=CritiqueResponse)
async def critique_pages(
    request: SummarizeRequest,
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    chunking_service: Annotated[ChunkingService, Depends(get_chunking_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    reranker_service: Annotated[RerankerService, Depends(get_reranker_service)],
    retrieval_cache: Annotated[RetrievalCache, Depends(get_retrieval_cache)],
    token_counter: Annotated[TokenCounter, Depends(get_token_counter)],
):
    """
    Review pinned pages for gaps, risks, and missing requirements.

    Uses the existing RAG retrieval pipeline but switches the answer style
    from summarization to critique.
    """
    try:
        request_started_at = time.perf_counter()
        settings = get_settings()
        prepared_pages = _prepare_pages(request.pages)
        logger.info(
            "Handling /api/critique pages=%s question=%s",
            len(prepared_pages),
            _question_excerpt(request.user_question),
        )
        if not prepared_pages:
            raise HTTPException(
                status_code=400,
                detail="At least one page with markdown content is required"
            )

        user_question = (
            (request.user_question or "").strip() or
            "Review these documents for missing requirements, edge cases, and risks."
        )

        retrieval_started_at = time.perf_counter()
        retrieved_chunks, retrieval_stats, source_conflicts = await retrieve_relevant_chunks(
            request=SummarizeRequest(
                pages=prepared_pages,
                user_question=user_question
            ),
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            vector_store=vector_store,
            reranker_service=reranker_service,
            retrieval_cache=retrieval_cache,
            token_counter=token_counter,
            reserve_tokens=6500,
            retrieval_mode="critique"
        )
        retrieval_latency_ms = round((time.perf_counter() - retrieval_started_at) * 1000)

        llm_chunks = retrieved_chunks
        hybrid_prefilter_stats: dict[str, object] = {"hybrid_prefilter_applied": False}
        if settings.hybrid_mode_enabled():
            llm_chunks, hybrid_prefilter_stats = await llm_service.prefilter_chunks(
                retrieved_chunks,
                user_question,
                task_kind="critique",
                planner_mode=str(retrieval_stats.get("planner_mode") or ""),
            )
            hybrid_prefilter_stats["hybrid_prefilter_context_tokens"] = token_counter.count_chunks(llm_chunks)

        generation_started_at = time.perf_counter()
        summary, issues, citations, token_usage = await llm_service.critique_from_chunks(
            chunks=llm_chunks,
            user_question=user_question
        )
        generation_latency_ms = round((time.perf_counter() - generation_started_at) * 1000)

        token_usage.update(retrieval_stats)
        token_usage.update(hybrid_prefilter_stats)
        token_usage["retrieval_latency_ms"] = retrieval_latency_ms
        token_usage.setdefault("generation_latency_ms", generation_latency_ms)
        token_usage["end_to_end_latency_ms"] = round((time.perf_counter() - request_started_at) * 1000)
        token_usage["issues_found"] = len(issues)
        token_usage["cost_estimate"] = token_counter.estimate_cost(
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0)
        )

        return CritiqueResponse(
            summary=summary,
            issues=issues,
            citations=citations,
            source_conflicts=source_conflicts or None,
            token_usage=token_usage,
            model_used=llm_service.get_model_name(),
        )

    except HTTPException as exc:
        log_method = logger.error if exc.status_code >= 500 else logger.warning
        log_method("Request failed path=/api/critique status=%s detail=%s", exc.status_code, exc.detail)
        raise
    except ValueError as e:
        logger.warning("Request validation failed path=/api/critique detail=%s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(
            "Unexpected critique error model=%s pages=%s question=%s",
            llm_service.get_model_name(),
            len(prepared_pages),
            _question_excerpt(request.user_question),
        )
        _raise_provider_http_error_if_needed(
            stage="chat",
            model_name=llm_service.get_model_name(),
            settings=settings,
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/generate-diagram", response_model=DiagramResponse)
async def generate_diagram(
    request: DiagramRequest,
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    chunking_service: Annotated[ChunkingService, Depends(get_chunking_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    reranker_service: Annotated[RerankerService, Depends(get_reranker_service)],
    retrieval_cache: Annotated[RetrievalCache, Depends(get_retrieval_cache)],
    token_counter: Annotated[TokenCounter, Depends(get_token_counter)],
):
    """
    Generate a Mermaid diagram from pinned pages using the RAG pipeline.

    - **pages**: List of pages with markdown content
    - **user_question**: Required question describing the desired visualization
    - **diagram_type**: Optional preferred Mermaid type
    """
    try:
        request_started_at = time.perf_counter()
        settings = get_settings()
        prepared_pages = _prepare_pages(request.pages)
        logger.info(
            "Handling /api/generate-diagram pages=%s question=%s diagram_type=%s",
            len(prepared_pages),
            _question_excerpt(request.user_question),
            request.diagram_type,
        )
        if not prepared_pages:
            raise HTTPException(
                status_code=400,
                detail="At least one page with markdown content is required"
            )

        user_question = (request.user_question or "").strip()

        if not user_question:
            raise HTTPException(
                status_code=400,
                detail="user_question is required for diagram generation"
            )

        retrieval_started_at = time.perf_counter()
        retrieved_chunks, retrieval_stats, source_conflicts = await retrieve_relevant_chunks(
            request=SummarizeRequest(
                pages=prepared_pages,
                user_question=user_question
            ),
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            vector_store=vector_store,
            reranker_service=reranker_service,
            retrieval_cache=retrieval_cache,
            token_counter=token_counter,
            reserve_tokens=7000,
            retrieval_mode="diagram"
        )
        retrieval_latency_ms = round((time.perf_counter() - retrieval_started_at) * 1000)

        llm_chunks = retrieved_chunks
        hybrid_prefilter_stats: dict[str, object] = {"hybrid_prefilter_applied": False}
        if settings.hybrid_mode_enabled():
            llm_chunks, hybrid_prefilter_stats = await llm_service.prefilter_chunks(
                retrieved_chunks,
                user_question,
                task_kind="diagram",
                planner_mode=str(retrieval_stats.get("planner_mode") or ""),
            )
            hybrid_prefilter_stats["hybrid_prefilter_context_tokens"] = token_counter.count_chunks(llm_chunks)

        generation_started_at = time.perf_counter()
        summary, mermaid_code, is_valid, diagram_type, citations, token_usage = (
            await llm_service.generate_diagram_from_chunks(
                chunks=llm_chunks,
                user_question=user_question,
                diagram_type=request.diagram_type
            )
        )
        generation_latency_ms = round((time.perf_counter() - generation_started_at) * 1000)

        token_usage.update(retrieval_stats)
        token_usage.update(hybrid_prefilter_stats)
        token_usage["retrieval_latency_ms"] = retrieval_latency_ms
        token_usage.setdefault("generation_latency_ms", generation_latency_ms)
        token_usage["end_to_end_latency_ms"] = round((time.perf_counter() - request_started_at) * 1000)
        token_usage["cost_estimate"] = token_counter.estimate_cost(
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0)
        )

        return DiagramResponse(
            summary=summary,
            mermaid_code=mermaid_code,
            is_valid=is_valid,
            diagram_type=diagram_type,
            citations=citations,
            source_conflicts=source_conflicts or None,
            token_usage=token_usage,
            model_used=llm_service.get_model_name(),
        )

    except HTTPException as exc:
        log_method = logger.error if exc.status_code >= 500 else logger.warning
        log_method("Request failed path=/api/generate-diagram status=%s detail=%s", exc.status_code, exc.detail)
        raise
    except ValueError as e:
        logger.warning("Request validation failed path=/api/generate-diagram detail=%s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(
            "Unexpected diagram error model=%s pages=%s question=%s diagram_type=%s",
            llm_service.get_model_name(),
            len(prepared_pages),
            _question_excerpt(request.user_question),
            request.diagram_type,
        )
        _raise_provider_http_error_if_needed(
            stage="chat",
            model_name=llm_service.get_model_name(),
            settings=settings,
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Annotated[Settings, Depends(get_settings)]):
    """
    Health check endpoint

    Returns current configuration and status.
    """
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        llm_provider=settings.llm_provider,
        model=settings.resolved_primary_model(),
    )
