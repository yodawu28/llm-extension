/**
 * API Client for Web Context Assistant Backend
 *
 * Communicates with FastAPI backend for page summarization using LLMs.
 */

// ============================================================================
// Type Definitions (matching backend Pydantic schemas)
// ============================================================================

export interface PageContent {
  title: string
  url: string
  markdown: string
  source_type: "confluence" | "jira" | "generic"
  metadata?: Record<string, any>
}

export interface SummarizeRequest {
  pages: PageContent[]
  user_question?: string
}

export interface DiagramRequest {
  pages: PageContent[]
  user_question: string
  diagram_type?: string
}

export interface Citation {
  page_title: string
  page_url: string
  source_type: string
}

export interface PageSuggestion {
  reason: string
  keywords: string[]
  confidence: "high" | "medium" | "low"
}

export interface CritiqueIssue {
  title: string
  severity: "high" | "medium" | "low"
  category: string
  evidence: string
  risk: string
  suggestion: string
  source_title?: string
}

export interface SummarizeResponse {
  summary: string
  citations: Citation[]
  suggestions?: PageSuggestion[]
  token_usage: {
    input_tokens: number
    output_tokens: number
    total_tokens: number
    chunk_stats?: {
      total_chunks: number
      total_chars: number
      avg_chunk_size: number
      pages_count: number
      chunks_per_page: number
    }
    retrieved_chunks?: number
    candidates_found?: number
    filtered_by_similarity?: number
    avg_similarity?: number
    selected_chunk_tokens?: number
    selected_pages?: number
    adjacent_chunks_added?: number
    reranker_used?: boolean
    reranker_model?: string
    reranker_candidates?: number
    input_pages?: number
    prepared_pages?: number
    deduped_pages?: number
    deduped_by_url?: number
    deduped_by_content?: number
    high_priority_pages?: number
    retrieval_mode?: string
    context_budget_tokens?: number
    retrieval_strategy?: string
    embedding_fallback_used?: boolean
    embedding_provider_forbidden?: boolean
    retrieval_no_match_fallback_used?: boolean
    routing_note?: string
    selected_sources_summary?: string
    omitted_sources_summary?: string
    cost_estimate?: {
      input_cost: number
      output_cost: number
      total_cost: number
      currency: string
    }
  }
  model_used: string
}

export interface CritiqueResponse {
  summary: string
  issues: CritiqueIssue[]
  citations: Citation[]
  token_usage: {
    input_tokens: number
    output_tokens: number
    total_tokens: number
    chunk_stats?: {
      total_chunks: number
      total_chars: number
      avg_chunk_size: number
      pages_count: number
      chunks_per_page: number
    }
    retrieved_chunks?: number
    candidates_found?: number
    filtered_by_similarity?: number
    avg_similarity?: number
    selected_chunk_tokens?: number
    selected_pages?: number
    adjacent_chunks_added?: number
    reranker_used?: boolean
    reranker_model?: string
    reranker_candidates?: number
    issues_found?: number
    input_pages?: number
    prepared_pages?: number
    deduped_pages?: number
    deduped_by_url?: number
    deduped_by_content?: number
    high_priority_pages?: number
    retrieval_mode?: string
    context_budget_tokens?: number
    retrieval_strategy?: string
    embedding_fallback_used?: boolean
    embedding_provider_forbidden?: boolean
    retrieval_no_match_fallback_used?: boolean
    routing_note?: string
    selected_sources_summary?: string
    omitted_sources_summary?: string
    cost_estimate?: {
      input_cost: number
      output_cost: number
      total_cost: number
      currency: string
    }
  }
  model_used: string
}

export interface DiagramResponse {
  summary: string
  mermaid_code: string
  is_valid: boolean
  diagram_type: string
  citations: Citation[]
  token_usage: {
    input_tokens: number
    output_tokens: number
    total_tokens: number
    chunk_stats?: {
      total_chunks: number
      total_chars: number
      avg_chunk_size: number
      pages_count: number
      chunks_per_page: number
    }
    retrieved_chunks?: number
    candidates_found?: number
    filtered_by_similarity?: number
    avg_similarity?: number
    selected_chunk_tokens?: number
    selected_pages?: number
    adjacent_chunks_added?: number
    reranker_used?: boolean
    reranker_model?: string
    reranker_candidates?: number
    input_pages?: number
    prepared_pages?: number
    deduped_pages?: number
    deduped_by_url?: number
    deduped_by_content?: number
    high_priority_pages?: number
    retrieval_mode?: string
    context_budget_tokens?: number
    retrieval_strategy?: string
    embedding_fallback_used?: boolean
    embedding_provider_forbidden?: boolean
    retrieval_no_match_fallback_used?: boolean
    routing_note?: string
    selected_sources_summary?: string
    omitted_sources_summary?: string
    cost_estimate?: {
      input_cost: number
      output_cost: number
      total_cost: number
      currency: string
    }
  }
  model_used: string
}

export interface HealthResponse {
  status: string
  app_name: string
  llm_provider: string
  model: string
}

export interface APIError {
  detail: string
  status: number
}

type APIPageSourceType = PageContent["source_type"]

function inferSourceType(url: string): APIPageSourceType {
  const normalizedUrl = url.toLowerCase()

  if (
    normalizedUrl.includes("/browse/") ||
    /jira\./i.test(normalizedUrl) ||
    /\/browse\/[A-Z]+-\d+/i.test(url)
  ) {
    return "jira"
  }

  if (
    normalizedUrl.includes("confluence") ||
    normalizedUrl.includes("atlassian.net/wiki") ||
    normalizedUrl.includes("/pages/viewpage.action") ||
    /wiki\./i.test(normalizedUrl)
  ) {
    return "confluence"
  }

  return "generic"
}

function normalizeString(value: unknown): string {
  if (typeof value === "string") {
    return value.trim()
  }

  if (value == null) {
    return ""
  }

  return String(value).trim()
}

function normalizeMetadata(value: unknown): Record<string, any> {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, any>)
    : {}
}

function fallbackTitle(title: string, url: string, index: number): string {
  if (title) {
    return title
  }

  if (url) {
    try {
      const parsed = new URL(url)
      const pathTail = parsed.pathname.split("/").filter(Boolean).pop()
      return pathTail || parsed.hostname || `Page ${index + 1}`
    } catch {
      return url
    }
  }

  return `Page ${index + 1}`
}

function fallbackUrl(url: string, title: string, index: number): string {
  if (url) {
    return url
  }

  const slug = (title || `page-${index + 1}`)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")

  return `local://${slug || `page-${index + 1}`}-${index + 1}`
}

function sanitizePages(pages: PageContent[]): PageContent[] {
  if (!Array.isArray(pages)) {
    return []
  }

  const sanitized: PageContent[] = []

  pages.forEach((page, index) => {
    const markdown = normalizeString(page?.markdown)

    if (!markdown) {
      return
    }

    const rawUrl = normalizeString(page?.url)
    const rawTitle = normalizeString(page?.title)
    const url = fallbackUrl(rawUrl, rawTitle, index)
    const title = fallbackTitle(rawTitle, url, index)
    const sourceType = normalizeString(page?.source_type) as APIPageSourceType

    sanitized.push({
      title,
      url,
      markdown,
      source_type:
        sourceType === "confluence" || sourceType === "jira" || sourceType === "generic"
          ? sourceType
          : inferSourceType(url),
      metadata: normalizeMetadata(page?.metadata)
    })
  })

  return sanitized
}

function normalizeBaseURL(url: string) {
  const normalized = normalizeString(url)
  return normalized.replace(/\/+$/, "") || "http://localhost:8000"
}

function resolveDefaultBaseURL() {
  return normalizeBaseURL(process.env.PLASMO_PUBLIC_API_BASE_URL || "http://localhost:8000")
}

// ============================================================================
// API Client Class
// ============================================================================

class APIClient {
  private baseURL: string
  private defaultRequestTimeoutMs: number
  private longRunningRequestTimeoutMs: number

  constructor(baseURL: string = resolveDefaultBaseURL()) {
    this.baseURL = normalizeBaseURL(baseURL)
    this.defaultRequestTimeoutMs = 90000
    this.longRunningRequestTimeoutMs = 120000
  }

  /**
   * Update base URL (useful for testing different endpoints)
   */
  setBaseURL(url: string) {
    this.baseURL = normalizeBaseURL(url)
  }

  private async request<T>(
    path: string,
    init: RequestInit,
    timeoutMs: number = this.defaultRequestTimeoutMs
  ): Promise<T> {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

    try {
      const response = await fetch(`${this.baseURL}${path}`, {
        ...init,
        signal: controller.signal
      })

      if (!response.ok) {
        throw await this.handleError(response)
      }

      return response.json()
    } catch (error: any) {
      if (error?.name === "AbortError") {
        throw {
          detail: `Request timed out after ${Math.round(timeoutMs / 1000)} seconds`,
          status: 408
        } as APIError
      }

      throw error
    } finally {
      clearTimeout(timeoutId)
    }
  }

  /**
   * Health check endpoint
   * GET /api/health
   */
  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/api/health", {
      method: "GET",
      headers: {
        "Content-Type": "application/json"
      }
    }, 5000)
  }

  /**
   * Summarize multiple pages (Phase 1: Basic summarization)
   * POST /api/summarize
   *
   * @param request - Summarization request with pages and optional question
   * @returns Summary with citations and token usage
   */
  async summarize(request: SummarizeRequest): Promise<SummarizeResponse> {
    const sanitizedRequest: SummarizeRequest = {
      pages: sanitizePages(request.pages),
      user_question: normalizeString(request.user_question) || undefined
    }

    if (sanitizedRequest.pages.length === 0) {
      throw {
        detail: "At least one page with markdown content is required",
        status: 400
      } as APIError
    }

    return this.request<SummarizeResponse>("/api/summarize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(sanitizedRequest)
    })
  }

  /**
   * RAG-based summarization (Phase 2: Chunking + Embedding + Retrieval)
   * POST /api/rag-summarize
   *
   * Requires user_question to be provided.
   * More efficient than basic summarize - only sends relevant chunks to LLM.
   *
   * @param request - Summarization request with pages and required question
   * @returns Summary with chunk-level citations and detailed stats
   */
  async ragSummarize(request: SummarizeRequest): Promise<SummarizeResponse> {
    const sanitizedRequest: SummarizeRequest = {
      pages: sanitizePages(request.pages),
      user_question: normalizeString(request.user_question) || undefined
    }

    if (sanitizedRequest.pages.length === 0) {
      throw {
        detail: "At least one page with markdown content is required",
        status: 400
      } as APIError
    }

    if (!sanitizedRequest.user_question) {
      throw {
        detail: "user_question is required for RAG-based summarization",
        status: 400
      } as APIError
    }

    return this.request<SummarizeResponse>("/api/rag-summarize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(sanitizedRequest)
    }, this.longRunningRequestTimeoutMs)
  }

  /**
   * Critique pinned pages for gaps, edge cases, and missing requirements.
   * POST /api/critique
   */
  async critique(request: SummarizeRequest): Promise<CritiqueResponse> {
    const sanitizedRequest: SummarizeRequest = {
      pages: sanitizePages(request.pages),
      user_question:
        normalizeString(request.user_question) ||
        "Review these documents for missing requirements, edge cases, and risks."
    }

    if (sanitizedRequest.pages.length === 0) {
      throw {
        detail: "At least one page with markdown content is required",
        status: 400
      } as APIError
    }

    return this.request<CritiqueResponse>("/api/critique", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(sanitizedRequest)
    }, this.longRunningRequestTimeoutMs)
  }

  /**
   * Generate Mermaid diagram from pinned pages using the RAG pipeline.
   * POST /api/generate-diagram
   */
  async generateDiagram(request: DiagramRequest): Promise<DiagramResponse> {
    const sanitizedRequest: DiagramRequest = {
      pages: sanitizePages(request.pages),
      user_question: normalizeString(request.user_question),
      diagram_type: normalizeString(request.diagram_type) || undefined
    }

    if (sanitizedRequest.pages.length === 0) {
      throw {
        detail: "At least one page with markdown content is required",
        status: 400
      } as APIError
    }

    if (!sanitizedRequest.user_question) {
      throw {
        detail: "user_question is required for diagram generation",
        status: 400
      } as APIError
    }

    return this.request<DiagramResponse>("/api/generate-diagram", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(sanitizedRequest)
    }, this.longRunningRequestTimeoutMs)
  }

  /**
   * Handle API errors
   */
  private async handleError(response: Response): Promise<APIError> {
    let detail = "Unknown error"

    try {
      const errorData = await response.json()
      detail = errorData.detail || errorData.message || detail
    } catch (e) {
      // If JSON parsing fails, use status text
      detail = response.statusText || detail
    }

    return {
      detail,
      status: response.status
    }
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const apiClient = new APIClient()

// Also export the class for testing/custom instances
export { APIClient }
