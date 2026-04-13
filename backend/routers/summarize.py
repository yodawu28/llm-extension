import re
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from config.settings import Settings, get_settings
from schemas.requests import (
    CritiqueResponse,
    DiagramRequest,
    DiagramResponse,
    HealthResponse,
    PageContent,
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

QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "show", "that", "the",
    "this", "to", "use", "what", "when", "where", "which", "who", "why", "with",
    "you", "your"
}


def get_llm_service() -> LLMService:
    """Dependency injection for LLM service"""
    return LLMService()


def get_chunking_service() -> ChunkingService:
    """Dependency injection for chunking service"""
    return ChunkingService(chunk_size=1000, chunk_overlap=200)


def get_embedding_service() -> EmbeddingService:
    """Dependency injection for embedding service"""
    settings = get_settings()
    return EmbeddingService(settings)


def get_vector_store() -> VectorStore:
    """Dependency injection for vector store"""
    return VectorStore(embedding_dimension=1536)


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


def _rerank_candidates(user_question: str, candidates: list[tuple]) -> list[tuple]:
    reranked = []

    for chunk, dense_score in candidates:
        hybrid_score = _calculate_hybrid_score(user_question, chunk, dense_score)
        reranked.append((chunk, hybrid_score))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked


def _chunk_key(chunk) -> tuple[str, int]:
    return (chunk.metadata.page_url, chunk.metadata.chunk_index)


def _expand_adjacent_chunks(
    selected_chunks: list,
    scored_chunks: list[tuple],
    all_chunks: list,
    token_counter: TokenCounter,
    reserve_tokens: int
) -> tuple[list, int]:
    if not selected_chunks:
        return selected_chunks, 0

    max_input = token_counter.settings.max_input_tokens - reserve_tokens
    total_tokens = token_counter.count_chunks(selected_chunks)
    selected_keys = {_chunk_key(chunk) for chunk in selected_chunks}
    chunk_lookup = {_chunk_key(chunk): chunk for chunk in all_chunks}

    adjacent_candidates: list[tuple[int, int, object]] = []

    for anchor_index, chunk in enumerate(selected_chunks):
        page_url = chunk.metadata.page_url
        chunk_index = chunk.metadata.chunk_index

        for offset_order, offset in enumerate((-1, 1)):
            neighbor_key = (page_url, chunk_index + offset)

            if neighbor_key in selected_keys or neighbor_key not in chunk_lookup:
                continue

            adjacent_candidates.append((anchor_index, offset_order, chunk_lookup[neighbor_key]))

    added_chunks = {}
    added_count = 0

    for _, _, neighbor in adjacent_candidates:
        neighbor_key = _chunk_key(neighbor)

        if neighbor_key in selected_keys or neighbor_key in added_chunks:
            continue

        neighbor_tokens = token_counter.count_text(neighbor.text)
        if total_tokens + neighbor_tokens > max_input:
            continue

        added_chunks[neighbor_key] = neighbor
        total_tokens += neighbor_tokens
        added_count += 1

    if not added_chunks:
        return selected_chunks, 0

    ordered_chunks = []
    seen_keys = set()

    for chunk in selected_chunks:
        page_url = chunk.metadata.page_url
        chunk_index = chunk.metadata.chunk_index

        prev_key = (page_url, chunk_index - 1)
        if prev_key in added_chunks and prev_key not in seen_keys:
            ordered_chunks.append(added_chunks[prev_key])
            seen_keys.add(prev_key)

        current_key = _chunk_key(chunk)
        if current_key not in seen_keys:
            ordered_chunks.append(chunk)
            seen_keys.add(current_key)

        next_key = (page_url, chunk_index + 1)
        if next_key in added_chunks and next_key not in seen_keys:
            ordered_chunks.append(added_chunks[next_key])
            seen_keys.add(next_key)

    for neighbor_key, neighbor in added_chunks.items():
        if neighbor_key not in seen_keys:
            ordered_chunks.append(neighbor)
            seen_keys.add(neighbor_key)

    return ordered_chunks, added_count


async def retrieve_relevant_chunks(
    request: SummarizeRequest,
    chunking_service: ChunkingService,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    reranker_service: RerankerService,
    retrieval_cache: RetrievalCache,
    token_counter: TokenCounter,
    reserve_tokens: int = 6000
):
    """Run the common RAG retrieval pipeline and return selected chunks plus metadata"""
    corpus_signature = retrieval_cache.build_corpus_signature(request.pages)
    cached_corpus = retrieval_cache.get_corpus(corpus_signature)
    corpus_cache_hit = cached_corpus is not None

    if cached_corpus is not None:
        chunks = cached_corpus.chunks
        chunk_stats = cached_corpus.chunk_stats
        embeddings = cached_corpus.embeddings
    else:
        chunks = chunking_service.chunk_pages(request.pages)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content to process - all pages are empty"
            )

        chunk_stats = chunking_service.get_chunk_stats(chunks)
        chunk_texts = [_build_chunk_embedding_text(chunk) for chunk in chunks]
        embeddings = await embedding_service.embed_texts(chunk_texts)

        retrieval_cache.set_corpus(
            corpus_signature,
            CachedCorpus(
                chunks=chunks,
                embeddings=embeddings,
                chunk_stats=chunk_stats
            )
        )

    vector_store.add_chunks(chunks, embeddings)

    question_signature = retrieval_cache.build_query_signature(request.user_question)
    question_embedding = retrieval_cache.get_query_embedding(question_signature)
    query_embedding_cache_hit = question_embedding is not None

    if question_embedding is None:
        question_embedding = await embedding_service.embed_single(request.user_question)
        retrieval_cache.set_query_embedding(question_signature, question_embedding)

    top_k_candidates = min(20, len(chunks))
    candidate_results = vector_store.search(question_embedding, top_k=top_k_candidates)

    if not candidate_results:
        raise HTTPException(
            status_code=400,
            detail="No relevant content found for the question"
        )

    reranked_results = _rerank_candidates(request.user_question, candidate_results)
    reranked_results, reranker_stats = reranker_service.rerank(
        request.user_question,
        reranked_results
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

    if not filtered_results:
        raise HTTPException(
            status_code=400,
            detail="No relevant content found for the question"
        )

    retrieved_chunks = token_counter.select_chunks_within_budget(
        chunks_with_scores=filtered_results,
        reserve_tokens=reserve_tokens,
        max_chunks_per_page=3
    )

    if not retrieved_chunks:
        raise HTTPException(
            status_code=400,
            detail="No chunks fit within token budget"
        )

    retrieved_chunks, adjacent_chunks_added = _expand_adjacent_chunks(
        selected_chunks=retrieved_chunks,
        scored_chunks=filtered_results,
        all_chunks=chunks,
        token_counter=token_counter,
        reserve_tokens=reserve_tokens
    )

    selected_tokens = token_counter.count_chunks(retrieved_chunks)
    selected_page_urls = {chunk.metadata.page_url for chunk in retrieved_chunks}
    similarity_scores = [
        score for chunk, score in filtered_results
        if chunk in retrieved_chunks
    ]

    retrieval_stats = {
        "chunk_stats": chunk_stats,
        "retrieved_chunks": len(retrieved_chunks),
        "candidates_found": len(candidate_results),
        "filtered_by_similarity": len(filtered_results),
        "avg_similarity": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
        "selected_chunk_tokens": selected_tokens,
        "selected_pages": len(selected_page_urls),
        "adjacent_chunks_added": adjacent_chunks_added,
        "corpus_cache_hit": corpus_cache_hit,
        "query_embedding_cache_hit": query_embedding_cache_hit,
        "similarity_threshold": min_similarity,
        "similarity_fallback_used": similarity_fallback_used
    }
    retrieval_stats.update(reranker_stats)

    return retrieved_chunks, retrieval_stats


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
        prepared_pages = _prepare_pages(request.pages)
        if not prepared_pages:
            raise HTTPException(
                status_code=400,
                detail="At least one page with markdown content is required"
            )

        user_question = (request.user_question or "").strip() or None

        # Validate total token count
        total_chars = sum(len(page.markdown) for page in prepared_pages)
        estimated_tokens = total_chars // 4  # Rough estimation

        settings = get_settings()
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

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
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
        prepared_pages = _prepare_pages(request.pages)
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

        retrieved_chunks, retrieval_stats = await retrieve_relevant_chunks(
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
            reserve_tokens=6000
        )

        # Step 7: Generate summary using only retrieved chunks
        summary, citations, token_usage, suggestions = await llm_service.summarize_from_chunks(
            chunks=retrieved_chunks,
            user_question=user_question
        )

        # Add chunking metadata to token usage
        token_usage.update(retrieval_stats)

        # Estimate API cost
        cost_estimate = token_counter.estimate_cost(
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0)
        )
        token_usage["cost_estimate"] = cost_estimate

        return SummarizeResponse(
            summary=summary,
            citations=citations,
            token_usage=token_usage,
            model_used=llm_service.get_model_name(),
            suggestions=suggestions
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
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
        prepared_pages = _prepare_pages(request.pages)
        if not prepared_pages:
            raise HTTPException(
                status_code=400,
                detail="At least one page with markdown content is required"
            )

        user_question = (
            (request.user_question or "").strip() or
            "Review these documents for missing requirements, edge cases, and risks."
        )

        retrieved_chunks, retrieval_stats = await retrieve_relevant_chunks(
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
            reserve_tokens=6500
        )

        summary, issues, citations, token_usage = await llm_service.critique_from_chunks(
            chunks=retrieved_chunks,
            user_question=user_question
        )

        token_usage.update(retrieval_stats)
        token_usage["issues_found"] = len(issues)
        token_usage["cost_estimate"] = token_counter.estimate_cost(
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0)
        )

        return CritiqueResponse(
            summary=summary,
            issues=issues,
            citations=citations,
            token_usage=token_usage,
            model_used=llm_service.get_model_name(),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
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
        prepared_pages = _prepare_pages(request.pages)
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

        retrieved_chunks, retrieval_stats = await retrieve_relevant_chunks(
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
            reserve_tokens=7000
        )

        summary, mermaid_code, is_valid, diagram_type, citations, token_usage = (
            await llm_service.generate_diagram_from_chunks(
                chunks=retrieved_chunks,
                user_question=user_question,
                diagram_type=request.diagram_type
            )
        )

        token_usage.update(retrieval_stats)
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
            token_usage=token_usage,
            model_used=llm_service.get_model_name(),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Annotated[Settings, Depends(get_settings)]):
    """
    Health check endpoint

    Returns current configuration and status.
    """
    model = settings.openai_model if settings.llm_provider == "openai" else settings.anthropic_model

    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        llm_provider=settings.llm_provider,
        model=model,
    )
