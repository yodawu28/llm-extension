import { useEffect, useRef, useState } from "react"

import "~style.css"

import type {
  CritiqueIssue,
  CritiqueResponse,
  DiagramResponse,
  PageContent as APIPageContent,
  SummarizeResponse
} from "~lib/api"
import { apiClient } from "~lib/api"
import { EXTENSION_BUILD_LABEL } from "~lib/build-info"
import type { LinkedResource } from "~lib/link-scanner"
import {
  DEFAULT_USER_SETTINGS,
  readUserSettings,
  updateUserSettings,
  type UserSettings
} from "~lib/user-settings"
import { MarkdownMessage } from "~components/MarkdownMessage"

interface ContextPage {
  id: string
  title: string
  url: string
  tokenEstimate: number
  addedAt: number
  favicon?: string
}

interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
  citations?: {
    page_title: string
    page_url: string
    source_type: string
  }[]
  suggestions?: {
    reason: string
    keywords: string[]
    confidence: "high" | "medium" | "low"
  }[]
  critiqueIssues?: CritiqueIssue[]
  diagnostics?: {
    title: string
    items: {
      label: string
      value: string
    }[]
  }
}

interface CurrentPageInfo {
  tabId: number
  title: string
  url: string
  hostname: string
  favicon?: string
}

interface BannerToast {
  title: string
  message: string
  tone: "info" | "success"
}

interface DiagramCacheEntry {
  cachedAt: number
  response: DiagramResponse
}

const DIAGRAM_CACHE_PREFIX = "diagram_cache_"
const DIAGRAM_CACHE_TTL_MS = 24 * 60 * 60 * 1000
const ASSISTANT_STREAM_CHUNK_SIZE = 24
const ASSISTANT_STREAM_DELAY_MS = 16

const SUGGESTION_STOP_WORDS = new Set([
  "about",
  "access",
  "beginner",
  "component",
  "components",
  "configuration",
  "consult",
  "control",
  "deploy",
  "details",
  "environment",
  "feature",
  "features",
  "for",
  "from",
  "guide",
  "implementation",
  "information",
  "instructions",
  "management",
  "missing",
  "page",
  "pages",
  "policy",
  "refer",
  "related",
  "setup",
  "specific",
  "steps",
  "system",
  "token",
  "tutorial",
  "tutorials",
  "usage",
  "use",
  "working"
])

const TITLE_MATCH_STOP_WORDS = new Set([
  "and",
  "for",
  "from",
  "into",
  "page",
  "spec",
  "task",
  "that",
  "the",
  "this",
  "with"
])

const USER_FACING_DIAGNOSTIC_LABELS = new Set([
  "Context focus",
  "Selected sources",
  "Ignored pages",
  "Routing note"
])

// Mock summary for testing
const MOCK_SUMMARY = `Based on the analyzed documents:

• **Authentication Flow**: OAuth 2.0 with JWT tokens, refresh every 15 minutes (Source: Auth PRD)

• **Database Schema**: PostgreSQL with 3 main tables - users, sessions, audit_logs (Source: Tech Spec)

• **API Rate Limiting**: 100 req/min per user, 500 for premium (Source: API Docs)

⚠️ **Contradiction**: Auth PRD says 30min timeout, but Tech Spec shows 60min`

function detectSourceType(url: string): "confluence" | "jira" | "generic" {
  if (
    url.includes("/browse/") ||
    /jira\./i.test(url) ||
    /\/browse\/[A-Z]+-\d+/i.test(url)
  ) {
    return "jira"
  }

  if (
    url.includes("confluence") ||
    url.includes("atlassian.net/wiki") ||
    url.includes("/pages/viewpage.action") ||
    /wiki\./i.test(url)
  ) {
    return "confluence"
  }

  return "generic"
}

function isDiagramRequest(question: string) {
  const normalized = question.toLowerCase()
  const keywords = [
    "diagram",
    "flowchart",
    "flow chart",
    "sequence",
    "architecture",
    "visualize",
    "visualise",
    "mermaid",
    "show me the flow",
    "show the flow",
    "draw",
    "graph"
  ]

  return keywords.some((keyword) => normalized.includes(keyword))
}

function isCritiqueRequest(question: string) {
  const normalized = question.toLowerCase()
  const keywords = [
    "review",
    "critique",
    "critic",
    "edge case",
    "edge-case",
    "missing requirement",
    "missing requirements",
    "gap",
    "gaps",
    "risk",
    "risks",
    "blind spot",
    "blind spots",
    "what's missing",
    "what is missing",
    "qa review",
    "failure mode",
    "failure modes"
  ]

  return keywords.some((keyword) => normalized.includes(keyword))
}

function buildDiagramMessageContent(summary: string, mermaidCode: string, isValid: boolean) {
  const parts = [summary.trim()]

  if (mermaidCode.trim()) {
    parts.push(`\`\`\`mermaid\n${mermaidCode.trim()}\n\`\`\``)
  }

  if (!isValid) {
    parts.push("⚠️ Mermaid validation did not fully pass. The raw code is still included below for inspection.")
  }

  return parts.filter(Boolean).join("\n\n")
}

function stripMissingInformationSection(content: string): string {
  return content
    .replace(/\n?---\s*\n?💡[\s\S]*$/m, "")
    .replace(/\n?💡\s*\*\*Missing Information:\*\*[\s\S]*$/m, "")
    .trim()
}

function extractSuggestionTerms(
  suggestion: NonNullable<ChatMessage["suggestions"]>[number]
): string[] {
  const rawTerms = [
    ...suggestion.keywords,
    ...suggestion.reason.split(/[^a-zA-Z0-9]+/)
  ]

  return Array.from(
    new Set(
      rawTerms
        .map((term) => term.trim().toLowerCase())
        .filter((term) => term.length >= 5 && !SUGGESTION_STOP_WORDS.has(term))
    )
  )
}

function normalizeTitleForMatching(value: string) {
  return value
    .toLowerCase()
    .replace(/[\[\]()"':]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
}

function normalizePageUrlForMatching(value: string) {
  try {
    const parsed = new URL(value)
    parsed.hash = ""
    parsed.search = ""
    return parsed.toString().replace(/\/+$/, "")
  } catch {
    return value.trim().replace(/\/+$/, "")
  }
}

function urlsLikelyMatch(left: string, right: string) {
  return normalizePageUrlForMatching(left) === normalizePageUrlForMatching(right)
}

function tokenizeTitleForMatching(value: string) {
  return normalizeTitleForMatching(value)
    .split(/[^a-z0-9]+/i)
    .map((part) => part.trim())
    .filter((part) => part.length >= 3 && !TITLE_MATCH_STOP_WORDS.has(part))
}

function titlesLikelyMatch(sourceTitle: string, currentTitle: string) {
  const normalizedSource = normalizeTitleForMatching(sourceTitle)
  const normalizedCurrent = normalizeTitleForMatching(currentTitle)

  if (!normalizedSource || !normalizedCurrent) {
    return false
  }

  if (
    normalizedSource.includes(normalizedCurrent) ||
    normalizedCurrent.includes(normalizedSource)
  ) {
    return true
  }

  const sourceTokens = new Set(tokenizeTitleForMatching(normalizedSource))
  const currentTokens = tokenizeTitleForMatching(normalizedCurrent)

  if (sourceTokens.size === 0 || currentTokens.length === 0) {
    return false
  }

  const overlapCount = currentTokens.filter((token) => sourceTokens.has(token)).length
  return overlapCount >= Math.max(2, Math.ceil(Math.min(sourceTokens.size, currentTokens.length) / 2))
}

function getSeverityRank(severity: CritiqueIssue["severity"]) {
  if (severity === "high") return 3
  if (severity === "medium") return 2
  return 1
}

function selectCritiqueIssuesForCurrentPage(
  issues: CritiqueIssue[],
  currentPageTitle: string
): CritiqueIssue[] {
  const issuesForCurrentPage = issues.filter((issue) => {
    if (!issue.source_title) {
      return true
    }

    return titlesLikelyMatch(issue.source_title, currentPageTitle)
  })

  const relevantIssues = issuesForCurrentPage.length > 0 ? issuesForCurrentPage : issues

  return [...relevantIssues].sort((left, right) => {
    const severityDelta = getSeverityRank(right.severity) - getSeverityRank(left.severity)

    if (severityDelta !== 0) {
      return severityDelta
    }

    const leftHasSourceMatch = left.source_title ? titlesLikelyMatch(left.source_title, currentPageTitle) : false
    const rightHasSourceMatch = right.source_title ? titlesLikelyMatch(right.source_title, currentPageTitle) : false

    if (leftHasSourceMatch !== rightHasSourceMatch) {
      return rightHasSourceMatch ? 1 : -1
    }

    return left.title.localeCompare(right.title)
  })
}

function filterSuggestionsForPinnedPages(
  suggestions: ChatMessage["suggestions"],
  pinnedPages: ContextPage[]
): ChatMessage["suggestions"] {
  if (!suggestions?.length) {
    return suggestions
  }

  const pinnedHaystacks = pinnedPages.map((page) =>
    `${page.title} ${page.url}`.toLowerCase()
  )

  return suggestions.filter((suggestion) => {
    const keywordPhrases = suggestion.keywords
      .map((keyword) => keyword.trim().toLowerCase())
      .filter((keyword) => keyword.length >= 5)
    const topicTerms = extractSuggestionTerms(suggestion)

    return !pinnedHaystacks.some((haystack) => {
      const phraseMatch = keywordPhrases.some((keyword) => haystack.includes(keyword))
      const tokenMatches = topicTerms.filter((term) => haystack.includes(term)).length
      return phraseMatch || tokenMatches >= 1
    })
  })
}

function buildRagDiagnostics(response: SummarizeResponse): ChatMessage["diagnostics"] | undefined {
  const usage = response.token_usage
  const stats = usage.chunk_stats
  const items: NonNullable<ChatMessage["diagnostics"]>["items"] = []

  if (stats) {
    items.push(
      { label: "Total chunks", value: String(stats.total_chunks) },
      { label: "Candidates", value: String(usage.candidates_found || 0) },
      { label: "Filtered", value: String(usage.filtered_by_similarity || 0) },
      { label: "Retrieved", value: String(usage.retrieved_chunks || 0) },
      { label: "Selected pages", value: String(usage.selected_pages || 0) },
      { label: "Adjacent added", value: String(usage.adjacent_chunks_added || 0) },
      { label: "Reranker", value: usage.reranker_used ? "cross-encoder" : "hybrid only" },
      { label: "Selected tokens", value: usage.selected_chunk_tokens?.toLocaleString() || "0" },
      { label: "Avg similarity", value: `${((usage.avg_similarity || 0) * 100).toFixed(1)}%` },
      { label: "Pages", value: String(stats.pages_count) }
    )
  }

  if (usage.reranker_candidates) {
    items.push({
      label: "Reranked candidates",
      value: String(usage.reranker_candidates)
    })
  }

  if (usage.retrieval_mode) {
    items.push({
      label: "Routing mode",
      value: usage.retrieval_mode
    })
  }

  if (usage.retrieval_strategy) {
    items.push({
      label: "Retrieval",
      value: usage.retrieval_strategy
    })
  }

  if (typeof usage.high_priority_pages === "number") {
    items.push({
      label: "High-priority pages",
      value: String(usage.high_priority_pages)
    })
  }

  if (typeof usage.deduped_pages === "number") {
    items.push({
      label: "Deduped pages",
      value: String(usage.deduped_pages)
    })
  }

  if (typeof usage.context_budget_tokens === "number") {
    items.push({
      label: "Context budget",
      value: usage.context_budget_tokens.toLocaleString()
    })
  }

  if (usage.embedding_fallback_used) {
    items.push({
      label: "Embedding fallback",
      value: usage.embedding_provider_forbidden ? "provider forbidden" : "used"
    })
  }

  if (usage.retrieval_no_match_fallback_used) {
    items.push({
      label: "Query fallback",
      value: "used top context"
    })
  }

  if (usage.selected_sources_summary) {
    items.push({
      label: "Selected sources",
      value: usage.selected_sources_summary
    })
  }

  if (usage.omitted_sources_summary) {
    items.push({
      label: "Ignored pages",
      value: usage.omitted_sources_summary
    })
  }

  if (usage.routing_note) {
    items.push({
      label: "Routing note",
      value: usage.routing_note
    })
  }

  if (usage.cost_estimate) {
    items.push({
      label: "Cost",
      value: `$${usage.cost_estimate.total_cost.toFixed(4)}`
    })
  }

  if (items.length === 0) return undefined

  return {
    title: "RAG Stats",
    items
  }
}

function getPageHostname(url: string) {
  try {
    return new URL(url).hostname
  } catch {
    return url
  }
}

function buildDiagramDiagnostics(response: DiagramResponse): ChatMessage["diagnostics"] | undefined {
  const usage = response.token_usage
  const stats = usage.chunk_stats
  const items: NonNullable<ChatMessage["diagnostics"]>["items"] = [
    { label: "Diagram type", value: response.diagram_type }
  ]

  if (stats) {
    items.push(
      { label: "Total chunks", value: String(stats.total_chunks) },
      { label: "Candidates", value: String(usage.candidates_found || 0) },
      { label: "Filtered", value: String(usage.filtered_by_similarity || 0) },
      { label: "Retrieved", value: String(usage.retrieved_chunks || 0) },
      { label: "Selected pages", value: String(usage.selected_pages || 0) },
      { label: "Adjacent added", value: String(usage.adjacent_chunks_added || 0) },
      { label: "Reranker", value: usage.reranker_used ? "cross-encoder" : "hybrid only" },
      { label: "Selected tokens", value: usage.selected_chunk_tokens?.toLocaleString() || "0" },
      { label: "Avg similarity", value: `${((usage.avg_similarity || 0) * 100).toFixed(1)}%` },
      { label: "Pages", value: String(stats.pages_count) }
    )
  }

  if (usage.reranker_candidates) {
    items.push({
      label: "Reranked candidates",
      value: String(usage.reranker_candidates)
    })
  }

  if (usage.retrieval_mode) {
    items.push({
      label: "Routing mode",
      value: usage.retrieval_mode
    })
  }

  if (usage.retrieval_strategy) {
    items.push({
      label: "Retrieval",
      value: usage.retrieval_strategy
    })
  }

  if (typeof usage.high_priority_pages === "number") {
    items.push({
      label: "High-priority pages",
      value: String(usage.high_priority_pages)
    })
  }

  if (typeof usage.deduped_pages === "number") {
    items.push({
      label: "Deduped pages",
      value: String(usage.deduped_pages)
    })
  }

  if (typeof usage.context_budget_tokens === "number") {
    items.push({
      label: "Context budget",
      value: usage.context_budget_tokens.toLocaleString()
    })
  }

  if (usage.embedding_fallback_used) {
    items.push({
      label: "Embedding fallback",
      value: usage.embedding_provider_forbidden ? "provider forbidden" : "used"
    })
  }

  if (usage.retrieval_no_match_fallback_used) {
    items.push({
      label: "Query fallback",
      value: "used top context"
    })
  }

  if (usage.selected_sources_summary) {
    items.push({
      label: "Selected sources",
      value: usage.selected_sources_summary
    })
  }

  if (usage.omitted_sources_summary) {
    items.push({
      label: "Ignored pages",
      value: usage.omitted_sources_summary
    })
  }

  if (usage.routing_note) {
    items.push({
      label: "Routing note",
      value: usage.routing_note
    })
  }

  if (usage.cost_estimate) {
    items.push({
      label: "Cost",
      value: `$${usage.cost_estimate.total_cost.toFixed(4)}`
    })
  }

  return {
    title: "Diagram Stats",
    items
  }
}

function withCacheBadge(
  diagnostics: ChatMessage["diagnostics"],
  status: "hit" | "miss"
): ChatMessage["diagnostics"] {
  if (!diagnostics) {
    return diagnostics
  }

  return {
    ...diagnostics,
    items: [
      { label: "Diagram cache", value: status === "hit" ? "hit" : "miss" },
      ...diagnostics.items
    ]
  }
}

function splitDiagnosticsForDisplay(diagnostics: NonNullable<ChatMessage["diagnostics"]>) {
  const whyItems = diagnostics.items.filter((item) => USER_FACING_DIAGNOSTIC_LABELS.has(item.label))
  const technicalItems = diagnostics.items.filter((item) => !USER_FACING_DIAGNOSTIC_LABELS.has(item.label))

  return {
    whyItems,
    technicalItems
  }
}

function isCurrentPageScopedRequest(question: string) {
  const normalized = question.trim().toLowerCase()

  if (!normalized) {
    return false
  }

  const directPhrases = [
    "this page",
    "current page",
    "this tab",
    "this document",
    "this doc",
    "this article",
    "this ticket",
    "this issue",
    "this prd",
    "this spec"
  ]

  return (
    directPhrases.some((phrase) => normalized.includes(phrase)) ||
    /\b(summarize|summarise|review|critique|explain|diagram|visualize|visualise)\b[\s\S]*\b(this|current)\s+(page|tab|document|doc|article|ticket|issue|prd|spec)\b/.test(normalized)
  )
}

function detectContextIntent(
  question: string,
  currentPageTitle?: string
): {
  mode: "default" | "current-only" | "compare-current-vs-pinned"
  rewrittenQuestion?: string
} {
  const normalized = question.trim().toLowerCase()
  const currentTitle = (currentPageTitle || "current page").trim()
  const hasCurrentPageReference = /\b(this|current)\s+(page|tab|document|doc|article|ticket|issue|prd|spec)\b/.test(normalized)

  const compareCurrentPage =
    /\b(compare|difference|different|diff|versus|vs\.?)\b/.test(normalized) &&
    hasCurrentPageReference

  const compareWithPinned =
    hasCurrentPageReference &&
    /\b(pinned|context basket|basket|other docs|other pages|reference docs|reference pages)\b/.test(normalized)

  if (compareCurrentPage || compareWithPinned) {
    return {
      mode: "compare-current-vs-pinned",
      rewrittenQuestion: `Compare the current page titled "${currentTitle}" against the other pinned pages. Highlight agreements, contradictions, missing context, and the most important differences.`
    }
  }

  const currentOnlyPhrases = [
    "only this page",
    "this page only",
    "only current page",
    "current page only",
    "only this doc",
    "this doc only",
    "only this document",
    "this document only",
    "critique this page only",
    "summarize only this page",
    "summarise only this page",
    "review this page only"
  ]

  if (currentOnlyPhrases.some((phrase) => normalized.includes(phrase)) || isCurrentPageScopedRequest(question)) {
    return {
      mode: "current-only"
    }
  }

  return {
    mode: "default"
  }
}

function prependContextFocusDiagnostic(
  diagnostics: ChatMessage["diagnostics"],
  focusLabel: string
): ChatMessage["diagnostics"] {
  if (!diagnostics || !focusLabel.trim()) {
    return diagnostics
  }

  return {
    ...diagnostics,
    items: [
      { label: "Context focus", value: focusLabel },
      ...diagnostics.items
    ]
  }
}

async function sha256Hex(value: string) {
  const encoded = new TextEncoder().encode(value)
  const digest = await crypto.subtle.digest("SHA-256", encoded)
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("")
}

async function buildDiagramCacheKey(question: string, pages: APIPageContent[]) {
  const normalizedPages = [...pages]
    .map((page) => ({
      title: page.title.trim(),
      url: page.url.trim(),
      markdown: page.markdown.trim(),
      source_type: page.source_type
    }))
    .sort((a, b) => `${a.url} ${a.title}`.localeCompare(`${b.url} ${b.title}`))

  const signature = JSON.stringify({
    question: question.trim().toLowerCase(),
    pages: normalizedPages
  })

  return `${DIAGRAM_CACHE_PREFIX}${await sha256Hex(signature)}`
}

async function readDiagramCache(cacheKey: string): Promise<DiagramResponse | null> {
  const result = await chrome.storage.local.get(cacheKey)
  const entry = result[cacheKey] as DiagramCacheEntry | undefined

  if (!entry) {
    return null
  }

  const age = Date.now() - entry.cachedAt
  if (age > DIAGRAM_CACHE_TTL_MS) {
    await chrome.storage.local.remove(cacheKey)
    return null
  }

  return entry.response
}

async function writeDiagramCache(cacheKey: string, response: DiagramResponse) {
  const entry: DiagramCacheEntry = {
    cachedAt: Date.now(),
    response
  }

  await chrome.storage.local.set({
    [cacheKey]: entry
  })
}

function buildCritiqueDiagnostics(response: CritiqueResponse): ChatMessage["diagnostics"] {
  const usage = response.token_usage
  const stats = usage.chunk_stats
  const items: NonNullable<ChatMessage["diagnostics"]>["items"] = [
    { label: "Issues found", value: String(response.issues.length) }
  ]

  if (stats) {
    items.push(
      { label: "Total chunks", value: String(stats.total_chunks) },
      { label: "Candidates", value: String(usage.candidates_found || 0) },
      { label: "Filtered", value: String(usage.filtered_by_similarity || 0) },
      { label: "Retrieved", value: String(usage.retrieved_chunks || 0) },
      { label: "Selected pages", value: String(usage.selected_pages || 0) },
      { label: "Adjacent added", value: String(usage.adjacent_chunks_added || 0) },
      { label: "Reranker", value: usage.reranker_used ? "cross-encoder" : "hybrid only" },
      { label: "Selected tokens", value: usage.selected_chunk_tokens?.toLocaleString() || "0" },
      { label: "Avg similarity", value: `${((usage.avg_similarity || 0) * 100).toFixed(1)}%` },
      { label: "Pages", value: String(stats.pages_count) }
    )
  }

  if (usage.cost_estimate) {
    items.push({
      label: "Cost",
      value: `$${usage.cost_estimate.total_cost.toFixed(4)}`
    })
  }

  if (usage.retrieval_mode) {
    items.push({
      label: "Routing mode",
      value: usage.retrieval_mode
    })
  }

  if (usage.retrieval_strategy) {
    items.push({
      label: "Retrieval",
      value: usage.retrieval_strategy
    })
  }

  if (typeof usage.high_priority_pages === "number") {
    items.push({
      label: "High-priority pages",
      value: String(usage.high_priority_pages)
    })
  }

  if (typeof usage.deduped_pages === "number") {
    items.push({
      label: "Deduped pages",
      value: String(usage.deduped_pages)
    })
  }

  if (typeof usage.context_budget_tokens === "number") {
    items.push({
      label: "Context budget",
      value: usage.context_budget_tokens.toLocaleString()
    })
  }

  if (usage.embedding_fallback_used) {
    items.push({
      label: "Embedding fallback",
      value: usage.embedding_provider_forbidden ? "provider forbidden" : "used"
    })
  }

  if (usage.retrieval_no_match_fallback_used) {
    items.push({
      label: "Query fallback",
      value: "used top context"
    })
  }

  if (usage.selected_sources_summary) {
    items.push({
      label: "Selected sources",
      value: usage.selected_sources_summary
    })
  }

  if (usage.omitted_sources_summary) {
    items.push({
      label: "Ignored pages",
      value: usage.omitted_sources_summary
    })
  }

  if (usage.routing_note) {
    items.push({
      label: "Routing note",
      value: usage.routing_note
    })
  }

  return {
    title: "Critique Stats",
    items
  }
}

function SidePanel() {
  // Current page state
  const [currentPage, setCurrentPage] = useState<CurrentPageInfo | null>(null)
  const [isCurrentPagePinned, setIsCurrentPagePinned] = useState(false)

  // Pinned pages (context basket)
  const [pinnedPages, setPinnedPages] = useState<ContextPage[]>([])
  const [totalTokens, setTotalTokens] = useState(0)
  const [showTokenWarning, setShowTokenWarning] = useState(false)

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingStage, setProcessingStage] = useState<string>("")

  // Mock mode for testing
  const [mockMode, setMockMode] = useState(false)

  // Animation state for pin icon
  const [isPinAnimating, setIsPinAnimating] = useState(false)
  const [showPinToast, setShowPinToast] = useState(false)
  const [bannerToast, setBannerToast] = useState<BannerToast | null>(null)
  const [isMaintenanceBusy, setIsMaintenanceBusy] = useState(false)
  const [isMindReaderScanBusy, setIsMindReaderScanBusy] = useState(false)
  const [userSettings, setUserSettings] = useState<UserSettings>(DEFAULT_USER_SETTINGS)
  const [highlightingCritiqueId, setHighlightingCritiqueId] = useState<string | null>(null)
  const [discoveredResources, setDiscoveredResources] = useState<LinkedResource[]>([])
  const [addingSuggestionUrl, setAddingSuggestionUrl] = useState<string | null>(null)
  const [showPinnedPagesList, setShowPinnedPagesList] = useState(false)
  const [pinnedPageFilter, setPinnedPageFilter] = useState("")
  const bannerToastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const streamTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const streamRunIdRef = useRef(0)
  const pinnedUrlSet = new Set(pinnedPages.map((page) => page.url))
  const availableDiscoveredResources = discoveredResources.filter(
    (resource) => !pinnedUrlSet.has(resource.url)
  )

  function showBannerToast(toast: BannerToast, duration = 4000) {
    setBannerToast(toast)

    if (bannerToastTimerRef.current) {
      clearTimeout(bannerToastTimerRef.current)
    }

    bannerToastTimerRef.current = setTimeout(() => {
      setBannerToast(null)
      bannerToastTimerRef.current = null
    }, duration)
  }

  function cancelAssistantStream() {
    if (streamTimerRef.current) {
      clearTimeout(streamTimerRef.current)
      streamTimerRef.current = null
    }

    streamRunIdRef.current += 1
  }

  function shouldStreamAssistantMessage(message: ChatMessage) {
    return Boolean(message.content.trim()) && !message.content.includes("```mermaid")
  }

  async function presentAssistantMessage(message: ChatMessage) {
    if (!shouldStreamAssistantMessage(message)) {
      setMessages((prev) => [...prev, message])
      return
    }

    cancelAssistantStream()
    const runId = streamRunIdRef.current
    const placeholderMessage: ChatMessage = {
      ...message,
      content: "",
      citations: undefined,
      suggestions: undefined,
      critiqueIssues: undefined,
      diagnostics: undefined
    }

    setMessages((prev) => [...prev, placeholderMessage])

    await new Promise<void>((resolve) => {
      let cursor = 0

      const tick = () => {
        if (runId !== streamRunIdRef.current) {
          resolve()
          return
        }

        cursor = Math.min(message.content.length, cursor + ASSISTANT_STREAM_CHUNK_SIZE)
        const nextContent = message.content.slice(0, cursor)

        setMessages((prev) =>
          prev.map((item) =>
            item.id === message.id
              ? {
                  ...item,
                  content: nextContent
                }
              : item
          )
        )

        if (cursor < message.content.length) {
          streamTimerRef.current = setTimeout(tick, ASSISTANT_STREAM_DELAY_MS)
          return
        }

        streamTimerRef.current = null
        setMessages((prev) => prev.map((item) => (item.id === message.id ? message : item)))
        resolve()
      }

      tick()
    })
  }

  // Load current page info
  useEffect(() => {
    loadCurrentPage()

    // Listen for tab changes from background script
    const handleMessage = (message: any) => {
      if (message.type === "ACTIVE_TAB_CHANGED" && message.payload) {
        setCurrentPage(message.payload)
        void loadDiscoveredResources(message.payload.tabId)
      }

      if (message.type === "LINKS_DISCOVERED" && message.payload?.count) {
        setDiscoveredResources(message.payload.resources || [])
        showBannerToast({
          title: "Mind Reader found related docs",
          message: `${message.payload.count} suggested pages are ready to review in this tab.`,
          tone: "info"
        })
      }

      if (message.type === "PAGES_ADDED" && message.payload?.count) {
        loadPinnedPages()
        loadCurrentPage()
        showBannerToast({
          title: "Pages added to context",
          message: `${message.payload.count} page${message.payload.count > 1 ? "s" : ""} added successfully.`,
          tone: "success"
        })
      }
    }

    chrome.runtime.onMessage.addListener(handleMessage)

    return () => {
      if (bannerToastTimerRef.current) {
        clearTimeout(bannerToastTimerRef.current)
        bannerToastTimerRef.current = null
      }
      cancelAssistantStream()
      chrome.runtime.onMessage.removeListener(handleMessage)
    }
  }, [])

  // Load pinned pages from storage
  useEffect(() => {
    loadPinnedPages()
  }, [])

  useEffect(() => {
    void loadUserSettings()
  }, [])

  // Update total tokens and check limits
  useEffect(() => {
    const total = pinnedPages.reduce((sum, page) => sum + page.tokenEstimate, 0)
    setTotalTokens(total)

    // Show warning if approaching token limit (80K tokens = ~320KB text)
    // Context window for GPT-4o is 128K, reserve 20K for output
    const TOKEN_WARNING_THRESHOLD = 80000
    setShowTokenWarning(total > TOKEN_WARNING_THRESHOLD)
  }, [pinnedPages])

  // Check if current page is pinned
  useEffect(() => {
    if (currentPage) {
      const isPinned = pinnedPages.some((p) => p.url === currentPage.url)
      setIsCurrentPagePinned(isPinned)
    }
  }, [currentPage, pinnedPages])

  async function loadCurrentPage() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
      if (tab?.id != null) {
        setCurrentPage({
          tabId: tab.id,
          title: tab.title || "Untitled",
          url: tab.url || "",
          hostname: new URL(tab.url || "").hostname,
          favicon: tab.favIconUrl
        })
        await loadDiscoveredResources(tab.id)
      } else {
        setDiscoveredResources([])
      }
    } catch (error) {
      console.error("Error loading current page:", error)
    }
  }

  async function loadDiscoveredResources(tabId?: number) {
    if (tabId == null || tabId < 0) {
      setDiscoveredResources([])
      return
    }

    const key = `discovered_links_${tabId}`
    const result = await chrome.storage.local.get(key)
    const discovered = result[key]
    setDiscoveredResources(discovered?.resources || [])
  }

  async function loadPinnedPages() {
    const result = await chrome.storage.local.get("pinnedPages")
    setPinnedPages(result.pinnedPages || [])
  }

  async function loadUserSettings() {
    try {
      setUserSettings(await readUserSettings())
    } catch (error) {
      console.error("Error loading user settings:", error)
    }
  }

  async function savePinnedPages(updatedPages: ContextPage[]) {
    await chrome.storage.local.set({ pinnedPages: updatedPages })
    setPinnedPages(updatedPages)
  }

  async function setMindReaderScanMode(nextMode: UserSettings["mindReaderScanMode"]) {
    try {
      const nextSettings = await updateUserSettings({
        mindReaderScanMode: nextMode
      })
      setUserSettings(nextSettings)
      showBannerToast({
        title:
          nextMode === "auto"
            ? "Mind Reader auto-scan enabled"
            : nextMode === "manual"
              ? "Mind Reader manual mode enabled"
              : "Mind Reader turned off",
        message:
          nextMode === "auto"
            ? "Related docs will be detected automatically on supported pages."
            : nextMode === "manual"
              ? "Mind Reader will scan only when you click Scan Now."
              : "Mind Reader scanning and prompts are disabled.",
        tone: "success"
      })
    } catch (error) {
      console.error("Error updating user settings:", error)
    }
  }

  async function toggleMindReaderPopups() {
    try {
      const nextSettings = await updateUserSettings({
        mindReaderPopupsEnabled: !userSettings.mindReaderPopupsEnabled
      })
      setUserSettings(nextSettings)
      showBannerToast({
        title: nextSettings.mindReaderPopupsEnabled ? "Mind Reader popups enabled" : "Mind Reader popups disabled",
        message: nextSettings.mindReaderPopupsEnabled
          ? "In-page import prompts and Chrome notifications are back on."
          : "Related docs will still be scanned, but popup prompts are now hidden.",
        tone: "success"
      })
    } catch (error) {
      console.error("Error updating user settings:", error)
    }
  }

  async function handleManualMindReaderScan() {
    if (!currentPage?.tabId || isMindReaderScanBusy || userSettings.mindReaderScanMode === "off") {
      return
    }

    setIsMindReaderScanBusy(true)

    try {
      const response = await chrome.runtime.sendMessage({
        action: "mind-reader-scan-current-tab",
        payload: {
          tabId: currentPage.tabId
        }
      })

      if (!response?.success) {
        throw new Error(response?.error || response?.reason || "Mind Reader scan failed")
      }

      const foundCount = Number(response.foundCount || 0)
      showBannerToast(
        foundCount > 0
          ? {
              title: "Mind Reader scan completed",
              message: `Found ${foundCount} related document${foundCount > 1 ? "s" : ""} for this page.`,
              tone: "success"
            }
          : {
              title: "Mind Reader found no new related docs",
              message: "This page was scanned, but no new suggestions were found.",
              tone: "info"
            },
        4500
      )

      await loadDiscoveredResources(currentPage.tabId)
    } catch (error) {
      console.error("Error running manual Mind Reader scan:", error)
      showBannerToast({
        title: "Could not scan this page",
        message: error instanceof Error ? error.message : "Mind Reader scan failed.",
        tone: "info"
      }, 5000)
    } finally {
      setIsMindReaderScanBusy(false)
    }
  }

  async function loadPageContentFromStorage(page: ContextPage): Promise<APIPageContent | null> {
    const storageKey = `page_${page.id}`
    const result = await chrome.storage.local.get(storageKey)
    const pageData = result[storageKey]
    const markdown = String(pageData?.markdown || "").trim()

    if (!markdown) {
      return null
    }

    return {
      title: page.title,
      url: page.url,
      markdown,
      source_type: detectSourceType(page.url),
      metadata: pageData?.metadata || {}
    }
  }

  async function scrapeCurrentPageForContext(): Promise<APIPageContent | null> {
    if (!currentPage?.tabId) {
      return null
    }

    try {
      const response = await chrome.tabs.sendMessage(currentPage.tabId, { action: "scrape-page" })

      if (!response?.success || !response.data) {
        return null
      }

      return {
        title: response.data.title,
        url: response.data.url,
        markdown: response.data.markdownContent,
        source_type: detectSourceType(response.data.url),
        metadata: response.data.metadata || {}
      }
    } catch (error) {
      console.warn("Could not scrape current page for scoped request:", error)
      return null
    }
  }

  async function togglePinCurrentPage() {
    if (!currentPage) return

    if (isCurrentPagePinned) {
      // Unpin
      const updated = pinnedPages.filter((p) => p.url !== currentPage.url)
      await savePinnedPages(updated)
      return
    }

    // Pin current page
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
      if (!tab.id) return

      // Try to scrape page
      const response = await chrome.tabs.sendMessage(tab.id, { action: "scrape-page" })

      if (response?.success && response.data) {
        const newPage: ContextPage = {
          id: crypto.randomUUID(),
          title: response.data.title,
          url: response.data.url,
          tokenEstimate: response.data.metadata.tokenEstimate,
          addedAt: Date.now(),
          favicon: tab.favIconUrl
        }

        // Store markdown separately
        await chrome.storage.local.set({
          [`page_${newPage.id}`]: {
            markdown: response.data.markdownContent,
            metadata: response.data.metadata
          }
        })

        const updated = [...pinnedPages, newPage]
        await savePinnedPages(updated)

        // Trigger animation and toast
        setIsPinAnimating(true)
        setShowPinToast(true)
        setTimeout(() => {
          setIsPinAnimating(false)
          setShowPinToast(false)
        }, 2000)
      } else {
        alert("⚠️ Could not extract content from this page.\n\nTry refreshing the page.")
      }
    } catch (error) {
      console.error("Error pinning page:", error)
      alert("⚠️ Please refresh this page (F5) and try again.")
    }
  }

  async function unpinPage(id: string) {
    const updated = pinnedPages.filter((p) => p.id !== id)
    await savePinnedPages(updated)
  }

  async function highlightCritiqueOnPage(
    message: ChatMessage,
    options?: {
      auto?: boolean
    }
  ) {
    if (!currentPage?.tabId || !message.critiqueIssues?.length) {
      if (!options?.auto) {
        showBannerToast({
          title: "No page available",
          message: "Open the source page in Chrome, then run highlight again.",
          tone: "info"
        })
      }
      return
    }

    setHighlightingCritiqueId(message.id)

    try {
      const rankedIssues = selectCritiqueIssuesForCurrentPage(message.critiqueIssues, currentPage.title)
      const issuesToHighlight = options?.auto ? rankedIssues.slice(0, 1) : rankedIssues

      const response = await chrome.tabs.sendMessage(currentPage.tabId, {
        action: "ghost-critic-highlight",
        payload: {
          issues: issuesToHighlight
        }
      })

      if (!response?.success) {
        throw new Error(response?.error || "Could not highlight critique findings")
      }

      const matchedIssues = Number(response.matchedIssues || 0)
      const totalIssues = Number(response.totalIssues || issuesToHighlight.length)

      if (matchedIssues > 0) {
        showBannerToast(
          options?.auto
            ? {
                title: "Ghost Critic auto-highlighted the top finding",
                message: `Highlighted the strongest matching issue on the current tab.`,
                tone: "success"
              }
            : {
                title: "Ghost Critic highlighted this page",
                message: `Matched ${matchedIssues} of ${totalIssues} finding${totalIssues > 1 ? "s" : ""} on the current tab.`,
                tone: "success"
              },
          4000
        )
      } else if (!options?.auto) {
        showBannerToast({
          title: "No matching text found",
          message: "Open the source page that contains the cited evidence, then try highlight again.",
          tone: "info"
        }, 5000)
      }
    } catch (error) {
      console.error("Error highlighting critique findings:", error)
      if (!options?.auto) {
        showBannerToast({
          title: "Could not highlight current page",
          message: "Refresh the tab once and try again. Chrome may not allow content scripts on this page.",
          tone: "info"
        }, 5000)
      }
    } finally {
      setHighlightingCritiqueId(null)
    }
  }

  async function clearCritiqueHighlights() {
    if (!currentPage?.tabId) {
      return
    }

    try {
      await chrome.tabs.sendMessage(currentPage.tabId, {
        action: "ghost-critic-clear-highlights"
      })
    } catch (error) {
      console.warn("Could not clear critique highlights:", error)
    }
  }

  function getMatchedSuggestionResources(
    suggestion: NonNullable<ChatMessage["suggestions"]>[number]
  ): LinkedResource[] {
    const searchTerms = [
      ...suggestion.keywords,
      ...suggestion.reason
        .split(/[^a-zA-Z0-9]+/)
        .map((part) => part.trim())
        .filter((part) => part.length >= 4)
    ].map((term) => term.toLowerCase())

    const scored = availableDiscoveredResources
      .map((resource) => {
        const haystack = `${resource.title} ${resource.url} ${resource.context}`.toLowerCase()
        const matchedTerms = Array.from(new Set(searchTerms.filter((term) => haystack.includes(term))))

        return {
          resource,
          score: matchedTerms.length,
          matchedTerms
        }
      })
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)

    return scored.slice(0, 3).map((item) => item.resource)
  }

  async function handleAddSuggestedResource(resource: LinkedResource) {
    if (addingSuggestionUrl === resource.url) return

    setAddingSuggestionUrl(resource.url)

    try {
      const response = await chrome.runtime.sendMessage({
        action: "mind-reader-add-all",
        payload: {
          resources: [resource],
          importId: crypto.randomUUID(),
          sourceTabId: currentPage?.tabId
        }
      })

      if (!response?.success) {
        throw new Error(response?.error || "Could not add suggested page")
      }

      if ((response.addedCount || 0) === 0) {
        throw new Error("This page is already in context")
      }

      await loadPinnedPages()
      showBannerToast(
        {
          title: "Suggested page added",
          message: `${resource.title} was added to context.`,
          tone: "success"
        },
        3500
      )
    } catch (error) {
      console.error("Error adding suggested resource:", error)
      alert(error instanceof Error ? error.message : "Could not add suggested page")
    } finally {
      setAddingSuggestionUrl(null)
    }
  }

  const normalizedPinnedFilter = pinnedPageFilter.trim().toLowerCase()
  const filteredPinnedPages = pinnedPages.filter((page) => {
    if (!normalizedPinnedFilter) {
      return true
    }

    const haystack = `${page.title} ${page.url} ${getPageHostname(page.url)}`.toLowerCase()
    return haystack.includes(normalizedPinnedFilter)
  })

  const groupedPinnedPages = Object.entries(
    filteredPinnedPages.reduce<Record<string, ContextPage[]>>((groups, page) => {
      const hostname = getPageHostname(page.url)
      groups[hostname] ||= []
      groups[hostname].push(page)
      return groups
    }, {})
  ).sort(([a], [b]) => a.localeCompare(b))

  async function handleMaintenanceAction(mode: "cache" | "all") {
    if (isMaintenanceBusy) return

    if (mode === "all") {
      const confirmed = window.confirm(
        "Reset all local extension data? This will clear pinned pages, imported page content, caches, and current chat history."
      )

      if (!confirmed) {
        return
      }
    }

    setIsMaintenanceBusy(true)

    try {
      const response = await chrome.runtime.sendMessage({
        action: mode === "cache" ? "clear-import-cache" : "reset-local-data"
      })

      if (!response?.success) {
        throw new Error(response?.error || "Maintenance action failed")
      }

      if (mode === "all") {
        cancelAssistantStream()
        setPinnedPages([])
        setMessages([])
        setIsCurrentPagePinned(false)
        setIsProcessing(false)
        setProcessingStage("")
        setShowTokenWarning(false)
      }

      showBannerToast(
        {
          title: mode === "cache" ? "Import cache cleared" : "Local extension data reset",
          message:
            response.removedCount > 0
              ? `Removed ${response.removedCount} stored item${response.removedCount > 1 ? "s" : ""}.`
              : "No stored items needed to be removed.",
          tone: "success"
        },
        4500
      )
    } catch (error) {
      console.error("Maintenance action failed:", error)
      alert(error instanceof Error ? error.message : "Maintenance action failed")
    } finally {
      setIsMaintenanceBusy(false)
    }
  }

  async function handleSendMessage() {
    if (!inputValue.trim() || isProcessing) return
    if (pinnedPages.length === 0) {
      alert("Please pin at least one page before asking questions")
      return
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: inputValue.trim(),
      timestamp: Date.now()
    }

    setMessages((prev) => [...prev, userMessage])
    const userQuestion = inputValue.trim()
    setInputValue("")
    setIsProcessing(true)

    try {
      if (mockMode) {
        // Mock mode: Use static response for testing UI
        await new Promise((resolve) => setTimeout(resolve, 1500))

        const assistantMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: MOCK_SUMMARY,
          timestamp: Date.now(),
          citations: [
            {
              page_title: "Authentication PRD v2.3",
              page_url: pinnedPages[0]?.url || "",
              source_type: "confluence"
            }
          ]
        }

        setProcessingStage("✍️ Streaming response...")
        await presentAssistantMessage(assistantMessage)
      } else {
        const diagramMode = isDiagramRequest(userQuestion)
        const critiqueMode = !diagramMode && isCritiqueRequest(userQuestion)
        const contextIntent = detectContextIntent(userQuestion, currentPage?.title)
        let diagramCacheKey: string | null = null
        let contextFocusLabel: string | null = null
        let effectiveUserQuestion = userQuestion

        // Real API mode: Call backend with Phase 2 RAG pipeline
        // 1. Fetch markdown content from storage for all pinned pages
        setProcessingStage("📄 Loading pages...")

        const pagesWithMarkdown: APIPageContent[] = (
          await Promise.all(pinnedPages.map((page) => loadPageContentFromStorage(page)))
        ).filter((page): page is APIPageContent => Boolean(page?.markdown.trim().length))

        if (pagesWithMarkdown.length === 0) {
          throw new Error("Pinned pages do not have imported content yet. Please pin or import them again.")
        }

        let requestPages = pagesWithMarkdown

        if (contextIntent.mode === "current-only" && currentPage) {
          const matchingPinnedPage = pagesWithMarkdown.find((page) => urlsLikelyMatch(page.url, currentPage.url))

          if (matchingPinnedPage) {
            requestPages = [matchingPinnedPage]
            contextFocusLabel = `Focused current page: ${matchingPinnedPage.title}`
          } else {
            setProcessingStage("📍 Reading current page...")
            const scrapedCurrentPage = await scrapeCurrentPageForContext()

            if (scrapedCurrentPage?.markdown.trim()) {
              requestPages = [scrapedCurrentPage]
              contextFocusLabel = `Focused live page: ${scrapedCurrentPage.title}`
            } else {
              contextFocusLabel = "Current page focus requested, but the active tab could not be read"
            }
          }
        } else if (contextIntent.mode === "compare-current-vs-pinned" && currentPage) {
          const matchingPinnedPage = pagesWithMarkdown.find((page) => urlsLikelyMatch(page.url, currentPage.url))
          const currentPageContent = matchingPinnedPage || await scrapeCurrentPageForContext()

          if (currentPageContent?.markdown.trim()) {
            const referencePages = pagesWithMarkdown.filter(
              (page) => !urlsLikelyMatch(page.url, currentPageContent.url)
            )

            if (referencePages.length > 0) {
              requestPages = [currentPageContent, ...referencePages]
              effectiveUserQuestion = contextIntent.rewrittenQuestion || userQuestion
              contextFocusLabel = `Comparing current page against ${referencePages.length} pinned page${referencePages.length > 1 ? "s" : ""}`
            } else {
              requestPages = [currentPageContent]
              contextFocusLabel = "Compare requested, but only the current page was available"
            }
          } else {
            contextFocusLabel = "Compare requested, but the current page could not be read"
          }
        }

        if (diagramMode) {
          try {
            diagramCacheKey = await buildDiagramCacheKey(effectiveUserQuestion, requestPages)
            setProcessingStage("⚡ Checking diagram cache...")
            const cachedDiagram = await readDiagramCache(diagramCacheKey)

            if (cachedDiagram) {
              const content = buildDiagramMessageContent(
                cachedDiagram.summary,
                cachedDiagram.mermaid_code,
                cachedDiagram.is_valid
              )

              const assistantMessage: ChatMessage = {
                id: crypto.randomUUID(),
                role: "assistant",
                content,
                timestamp: Date.now(),
                citations: cachedDiagram.citations,
                diagnostics: prependContextFocusDiagnostic(
                  withCacheBadge(buildDiagramDiagnostics(cachedDiagram), "hit"),
                  contextFocusLabel || ""
                )
              }

              setMessages((prev) => [...prev, assistantMessage])
              return
            }
          } catch (error) {
            console.warn("Diagram cache read failed:", error)
            diagramCacheKey = null
          }
        }

        // 2. Call backend with detailed progress
        setProcessingStage("🩺 Checking backend...")
        await apiClient.healthCheck()

        setProcessingStage("🔪 Chunking documents...")
        setProcessingStage("🧬 Creating embeddings...")
        setProcessingStage("🔍 Searching relevant content...")
        if (diagramMode) {
          const response = await apiClient.generateDiagram({
            pages: requestPages,
            user_question: effectiveUserQuestion
          })

          try {
            const cacheKey = diagramCacheKey || await buildDiagramCacheKey(effectiveUserQuestion, requestPages)
            await writeDiagramCache(cacheKey, response)
          } catch (error) {
            console.warn("Diagram cache write failed:", error)
          }

          setProcessingStage("🧭 Building Mermaid diagram...")
          let content = buildDiagramMessageContent(
            response.summary,
            response.mermaid_code,
            response.is_valid
          )

          const assistantMessage: ChatMessage = {
            id: crypto.randomUUID(),
            role: "assistant",
            content,
            timestamp: Date.now(),
            citations: response.citations,
            diagnostics: prependContextFocusDiagnostic(
              withCacheBadge(buildDiagramDiagnostics(response), "miss"),
              contextFocusLabel || ""
            )
          }

          setMessages((prev) => [...prev, assistantMessage])
        } else if (critiqueMode) {
          setProcessingStage("🕳️ Looking for gaps...")
          const response = await apiClient.critique({
            pages: requestPages,
            user_question: effectiveUserQuestion
          })

          setProcessingStage("🧪 Writing review...")
          const assistantMessage: ChatMessage = {
            id: crypto.randomUUID(),
            role: "assistant",
            content: response.summary,
            timestamp: Date.now(),
            citations: response.citations,
            critiqueIssues: response.issues,
            diagnostics: prependContextFocusDiagnostic(
              buildCritiqueDiagnostics(response),
              contextFocusLabel || ""
            )
          }

          setProcessingStage("✍️ Streaming review...")
          await presentAssistantMessage(assistantMessage)

          if (currentPage) {
            setProcessingStage("🎯 Auto-highlighting top finding...")
            await highlightCritiqueOnPage(assistantMessage, { auto: true })
          }
        } else {
          const response = await apiClient.ragSummarize({
            pages: requestPages,
            user_question: effectiveUserQuestion
          })

          setProcessingStage("🤔 Thinking...")
          // 3. Display assistant response with chunk stats
          const filteredSuggestions = filterSuggestionsForPinnedPages(
            response.suggestions,
            pinnedPages
          )
          let content = stripMissingInformationSection(response.summary)

          const assistantMessage: ChatMessage = {
            id: crypto.randomUUID(),
            role: "assistant",
            content,
            timestamp: Date.now(),
            citations: response.citations,
            suggestions: filteredSuggestions,
            diagnostics: prependContextFocusDiagnostic(
              buildRagDiagnostics(response),
              contextFocusLabel || ""
            )
          }

          setProcessingStage("✍️ Streaming response...")
          await presentAssistantMessage(assistantMessage)
        }
      }
    } catch (error: any) {
      console.error("Error getting response:", error)

      // Display error message to user
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `❌ **Error:** ${error.detail || error.message || "Could not get response from backend."}\n\n${
          error.status === 0 || error.message?.includes("fetch")
            ? "**Possible causes:**\n• Backend server is not running\n• Backend is not accessible at http://localhost:8000\n\n**To fix:**\n1. Start backend: `cd backend && source venv/bin/activate && uvicorn main:app --reload`\n2. Check backend is running at http://localhost:8000/api/health"
            : error.status === 408 || error.message?.toLowerCase?.().includes("timed out")
              ? "**Possible causes:**\n• Backend request is hanging on embeddings or LLM call\n• API provider is slow or unreachable\n\n**To fix:**\n1. Check backend terminal logs\n2. Verify `backend/.env` has valid provider credentials (`OPENAI_API_KEY` or `PAT_TOKEN + AWS_GATEWAY_URL`)\n3. Try a smaller set of pinned pages and retry"
            : (error.detail || error.message || "").toLowerCase().includes("no relevant content found")
              ? "**Possible causes:**\n• The question wording does not overlap enough with the pinned pages\n• The pinned pages are related, but not explicit enough for retrieval\n\n**To fix:**\n1. Rephrase the question with terms that appear in the documents\n2. Pin a more direct PRD/spec/page for this topic\n3. Retry once after narrowing the context basket"
            : (error.detail || error.message || "").toLowerCase().includes("do not contain enough overlapping context")
              ? "**Possible causes:**\n• The current pinned pages do not cover this topic well enough\n• Retrieval could not find a confident match in the available context\n\n**To fix:**\n1. Pin a more relevant page for this topic\n2. Ask a narrower question using the document's own terminology\n3. Remove unrelated pages and retry"
            : (error.detail || error.message || "").toLowerCase().includes("do not contain enough usable context")
              ? "**Possible causes:**\n• The pinned pages are too weakly related to the question\n• Most of the current basket is noise for this request\n\n**To fix:**\n1. Remove unrelated pages from the context basket\n2. Pin a more direct PRD/spec/page for this topic\n3. Rephrase the question using the source document's wording"
            : error.status === 400
              ? "**Possible causes:**\n• Input too large (exceeded token limit)\n• The current request or pinned context is not specific enough"
              : (error.detail || error.message || "").toLowerCase().includes("gateway forbids the chat model")
                ? "**Possible causes:**\n• `OPENAI_MODEL` is not on the gateway allow-list\n• The PAT has access to the gateway but not to this chat provider/model\n\n**To fix:**\n1. Check `backend/.env` for `OPENAI_MODEL`\n2. Switch to a model explicitly allowed by the gateway\n3. Restart backend after updating `.env`"
              : (error.detail || error.message || "").toLowerCase().includes("selected provider is forbidden")
                ? "**Possible causes:**\n• `PAT_TOKEN` is valid but the selected model/provider is blocked by the gateway\n• The gateway allows chat but denies embeddings for the configured model/provider\n\n**To fix:**\n1. Check `backend/.env` has the correct `PAT_TOKEN` and `AWS_GATEWAY_URL`\n2. Verify the configured `OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` are allowed by the gateway\n3. Check backend logs to see whether the failure came from chat or embeddings"
              : error.status === 500
                ? "**Possible causes:**\n• Provider credentials are missing or invalid in `backend/.env`\n• Internal gateway configuration is incomplete (`PAT_TOKEN`, `AWS_GATEWAY_URL`, model names)\n• Backend internal error\n\n**To fix:**\n1. Check backend/.env has `OPENAI_API_KEY` or `PAT_TOKEN + AWS_GATEWAY_URL`\n2. Verify the configured model names are allowed by your gateway\n3. Check backend logs for errors"
                : ""
        }`,
        timestamp: Date.now()
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsProcessing(false)
      setProcessingStage("")
    }
  }

  function handleKeyPress(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="plasmo-w-full plasmo-h-screen plasmo-bg-gray-50 plasmo-flex plasmo-flex-col plasmo-relative">
      {/* Toast Notification */}
      {showPinToast && (
        <div className="plasmo-absolute plasmo-top-4 plasmo-left-1/2 plasmo-transform plasmo--translate-x-1/2 plasmo-z-50 plasmo-bg-green-600 plasmo-text-white plasmo-px-4 plasmo-py-2 plasmo-rounded-lg plasmo-shadow-lg plasmo-flex plasmo-items-center plasmo-gap-2 plasmo-animate-bounce">
          <span className="plasmo-text-lg">✓</span>
          <span className="plasmo-text-sm plasmo-font-medium">Page pinned!</span>
        </div>
      )}

      {bannerToast && (
        <div
          className={`plasmo-absolute plasmo-top-16 plasmo-left-1/2 plasmo-transform plasmo--translate-x-1/2 plasmo-z-50 plasmo-max-w-md plasmo-px-4 plasmo-py-3 plasmo-rounded-lg plasmo-shadow-lg plasmo-border plasmo-flex plasmo-items-start plasmo-gap-3 ${
            bannerToast.tone === "success"
              ? "plasmo-bg-green-50 plasmo-border-green-200 plasmo-text-green-900"
              : "plasmo-bg-blue-50 plasmo-border-blue-200 plasmo-text-blue-900"
          }`}
        >
          <span className="plasmo-text-lg plasmo-leading-none">
            {bannerToast.tone === "success" ? "✓" : "💡"}
          </span>
          <div className="plasmo-min-w-0">
            <div className="plasmo-text-sm plasmo-font-semibold">{bannerToast.title}</div>
            <div className="plasmo-text-xs plasmo-mt-1">{bannerToast.message}</div>
          </div>
          <button
            onClick={() => setBannerToast(null)}
            className="plasmo-text-sm plasmo-opacity-60 hover:plasmo-opacity-100"
          >
            ×
          </button>
        </div>
      )}

      {/* Current Page Header */}
      <header className="plasmo-bg-white plasmo-border-b plasmo-border-gray-200 plasmo-p-3">
        <div className="plasmo-flex plasmo-items-start plasmo-justify-between plasmo-gap-3">
          <div className="plasmo-flex plasmo-items-start plasmo-gap-2 plasmo-flex-1 plasmo-min-w-0">
            {currentPage?.favicon && (
              <img
                src={currentPage.favicon}
                alt=""
                className="plasmo-w-4 plasmo-h-4 plasmo-mt-1 plasmo-flex-shrink-0"
              />
            )}
            <div className="plasmo-flex-1 plasmo-min-w-0">
              <div className="plasmo-text-sm plasmo-font-semibold plasmo-text-gray-800 plasmo-truncate">
                {currentPage?.title || "No page"}
              </div>
              <div className="plasmo-flex plasmo-items-center plasmo-gap-2 plasmo-min-w-0">
                <div className="plasmo-text-xs plasmo-text-gray-500 plasmo-truncate">
                  {currentPage?.hostname || ""}
                </div>
                <span className="plasmo-flex-shrink-0 plasmo-rounded-full plasmo-border plasmo-border-sky-200 plasmo-bg-sky-50 plasmo-px-2 plasmo-py-0.5 plasmo-text-[10px] plasmo-font-semibold plasmo-uppercase plasmo-tracking-[0.08em] plasmo-text-sky-700">
                  {EXTENSION_BUILD_LABEL}
                </span>
              </div>
            </div>
          </div>

          <button
            onClick={togglePinCurrentPage}
            disabled={!currentPage}
            className={`plasmo-text-2xl plasmo-p-2 plasmo-rounded-lg plasmo-transition-all plasmo-duration-200 ${
              isCurrentPagePinned
                ? "plasmo-bg-primary-100 plasmo-ring-2 plasmo-ring-primary-500 plasmo-shadow-md plasmo-scale-110 plasmo-rotate-12"
                : "plasmo-bg-gray-50 plasmo-opacity-50 hover:plasmo-opacity-100 hover:plasmo-bg-gray-100 hover:plasmo-scale-105"
            } ${
              isPinAnimating ? "plasmo-animate-bounce" : ""
            } disabled:plasmo-opacity-20 disabled:plasmo-cursor-not-allowed disabled:plasmo-bg-transparent`}
            title={isCurrentPagePinned ? "Unpin this page" : "Pin this page"}
          >
            📌
          </button>
        </div>

        {/* Settings */}
        <details className="plasmo-mt-2 plasmo-text-xs">
          <summary className="plasmo-text-gray-500 plasmo-cursor-pointer hover:plasmo-text-gray-700">
            ⚙️ Settings
          </summary>
          <div className="plasmo-mt-2 plasmo-space-y-3">
            <div className="plasmo-rounded-lg plasmo-border plasmo-border-slate-200 plasmo-bg-slate-50 plasmo-p-3">
              <div className="plasmo-flex plasmo-items-start plasmo-justify-between plasmo-gap-3">
                <div>
                  <div className="plasmo-text-xs plasmo-font-semibold plasmo-text-slate-800">
                    Mind Reader scan mode
                  </div>
                  <div className="plasmo-mt-1 plasmo-text-[11px] plasmo-leading-relaxed plasmo-text-slate-500">
                    Auto scans on page load. Manual scans only when you click the button. Off disables Mind Reader entirely.
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => void handleManualMindReaderScan()}
                  disabled={isMindReaderScanBusy || userSettings.mindReaderScanMode === "off" || !currentPage}
                  className="plasmo-rounded-md plasmo-border plasmo-border-blue-200 plasmo-bg-blue-50 plasmo-px-2.5 plasmo-py-1 plasmo-text-[11px] plasmo-font-semibold plasmo-text-blue-700 disabled:plasmo-cursor-not-allowed disabled:plasmo-opacity-50"
                >
                  {isMindReaderScanBusy ? "Scanning..." : "Scan Now"}
                </button>
              </div>
              <div className="plasmo-mt-3 plasmo-grid plasmo-grid-cols-3 plasmo-gap-2">
                {(["auto", "manual", "off"] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => void setMindReaderScanMode(mode)}
                    className={`plasmo-rounded-md plasmo-border plasmo-px-2 plasmo-py-1.5 plasmo-text-[11px] plasmo-font-semibold plasmo-capitalize ${
                      userSettings.mindReaderScanMode === mode
                        ? "plasmo-border-primary-500 plasmo-bg-primary-600 plasmo-text-white"
                        : "plasmo-border-slate-200 plasmo-bg-white plasmo-text-slate-600"
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>

            <label className="plasmo-flex plasmo-items-start plasmo-justify-between plasmo-gap-3 plasmo-rounded-lg plasmo-border plasmo-border-slate-200 plasmo-bg-slate-50 plasmo-p-3">
              <div>
                <div className="plasmo-text-xs plasmo-font-semibold plasmo-text-slate-800">
                  Mind Reader popup import
                </div>
                <div className="plasmo-mt-1 plasmo-text-[11px] plasmo-leading-relaxed plasmo-text-slate-500">
                  Show in-page import prompts and Chrome notifications when related docs are found.
                </div>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={userSettings.mindReaderPopupsEnabled}
                onClick={() => void toggleMindReaderPopups()}
                disabled={userSettings.mindReaderScanMode === "off"}
                className={`plasmo-inline-flex plasmo-h-6 plasmo-w-11 plasmo-flex-shrink-0 plasmo-items-center plasmo-rounded-full plasmo-border plasmo-transition ${
                  userSettings.mindReaderPopupsEnabled
                    ? "plasmo-border-primary-500 plasmo-bg-primary-600"
                    : "plasmo-border-slate-300 plasmo-bg-slate-300"
                } disabled:plasmo-cursor-not-allowed disabled:plasmo-opacity-50`}
              >
                <span
                  className={`plasmo-inline-block plasmo-h-5 plasmo-w-5 plasmo-rounded-full plasmo-bg-white plasmo-shadow-sm plasmo-transition-transform ${
                    userSettings.mindReaderPopupsEnabled ? "plasmo-translate-x-5" : "plasmo-translate-x-0.5"
                  }`}
                />
              </button>
            </label>

            <div className="plasmo-flex plasmo-flex-wrap plasmo-gap-2">
              <button
                onClick={() => setMockMode(!mockMode)}
                className={`plasmo-px-2 plasmo-py-1 plasmo-rounded plasmo-text-xs ${
                  mockMode
                    ? "plasmo-bg-green-100 plasmo-text-green-800"
                    : "plasmo-bg-gray-100 plasmo-text-gray-600"
                }`}
              >
                Mock: {mockMode ? "ON" : "OFF"}
              </button>
              <button
                onClick={() => void handleMaintenanceAction("cache")}
                disabled={isMaintenanceBusy}
                className="plasmo-px-2 plasmo-py-1 plasmo-rounded plasmo-text-xs plasmo-bg-amber-50 plasmo-text-amber-800 disabled:plasmo-opacity-50"
              >
                Clear Cache
              </button>
              <button
                onClick={() => void handleMaintenanceAction("all")}
                disabled={isMaintenanceBusy}
                className="plasmo-px-2 plasmo-py-1 plasmo-rounded plasmo-text-xs plasmo-bg-rose-50 plasmo-text-rose-800 disabled:plasmo-opacity-50"
              >
                Reset Local Data
              </button>
            </div>
          </div>
          <div className="plasmo-mt-2 plasmo-text-[11px] plasmo-leading-relaxed plasmo-text-gray-500">
            `Clear Cache` removes fetched/import discovery cache and saved diagram results.
            `Reset Local Data` also clears pinned pages, stored markdown, and current chat state in this panel.
          </div>
        </details>
      </header>

      {/* Chat Area */}
      <main className="plasmo-flex-1 plasmo-overflow-y-auto plasmo-p-4 plasmo-space-y-4">
        {messages.length === 0 ? (
          <div className="plasmo-flex plasmo-flex-col plasmo-items-center plasmo-justify-center plasmo-h-full plasmo-text-center">
            <div className="plasmo-text-4xl plasmo-mb-3">👋</div>
            <div className="plasmo-text-lg plasmo-font-semibold plasmo-text-gray-800 plasmo-mb-1">
              Xin chào!
            </div>
            <div className="plasmo-text-sm plasmo-text-gray-600">
              Pin bất kỳ trang web nào và hỏi mình nhé
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`plasmo-flex ${msg.role === "user" ? "plasmo-justify-end" : "plasmo-justify-start"}`}
            >
              <div className="plasmo-max-w-[85%]">
                <div
                  className={`plasmo-rounded-lg plasmo-p-3 ${
                    msg.role === "user"
                      ? "plasmo-bg-primary-600 plasmo-text-white"
                      : "plasmo-bg-white plasmo-border plasmo-border-gray-200 plasmo-text-gray-800"
                  }`}
                >
                  {msg.role === "user" ? (
                    <div className="plasmo-text-sm plasmo-whitespace-pre-wrap plasmo-leading-relaxed">
                      {msg.content}
                    </div>
                  ) : (
                    <MarkdownMessage content={msg.content} citations={msg.citations} />
                  )}

                  {msg.citations && msg.citations.length > 0 && (
                    <div className="plasmo-mt-3 plasmo-pt-3 plasmo-border-t plasmo-border-gray-200">
                      <div className="plasmo-text-xs plasmo-font-semibold plasmo-text-gray-700 plasmo-mb-2">
                        📚 Sources:
                      </div>
                      <div className="plasmo-space-y-1">
                        {msg.citations.map((citation, idx) => (
                          <a
                            key={idx}
                            href={citation.page_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="plasmo-block plasmo-text-xs plasmo-text-primary-600 hover:plasmo-underline"
                          >
                            {idx + 1}. {citation.page_title}
                          </a>
                        ))}
                      </div>
                    </div>
                  )}

                  {msg.suggestions && msg.suggestions.length > 0 && (
                    <div className="plasmo-mt-3 plasmo-pt-3 plasmo-border-t plasmo-border-gray-200">
                      <div className="plasmo-text-xs plasmo-font-semibold plasmo-text-gray-700 plasmo-mb-2">
                        💡 Suggested Pages to Pin:
                      </div>
                      <div className="plasmo-space-y-2">
                        {msg.suggestions.map((suggestion, idx) => (
                          <div
                            key={idx}
                            className="plasmo-bg-blue-50 plasmo-border plasmo-border-blue-200 plasmo-rounded plasmo-p-2"
                          >
                            <div className="plasmo-text-xs plasmo-text-gray-800 plasmo-mb-1">
                              {suggestion.reason}
                            </div>
                            <div className="plasmo-flex plasmo-flex-wrap plasmo-gap-1">
                              {suggestion.keywords.map((keyword, kidx) => (
                                <span
                                  key={kidx}
                                  className="plasmo-text-xs plasmo-bg-blue-100 plasmo-text-blue-800 plasmo-px-2 plasmo-py-0.5 plasmo-rounded-full"
                                >
                                  {keyword}
                                </span>
                              ))}
                            </div>

                            {getMatchedSuggestionResources(suggestion).length > 0 && (
                              <div className="plasmo-mt-2 plasmo-space-y-2">
                                {getMatchedSuggestionResources(suggestion).map((resource) => (
                                  <div
                                    key={resource.url}
                                    className="plasmo-rounded-md plasmo-border plasmo-border-blue-100 plasmo-bg-white plasmo-p-2"
                                  >
                                    <a
                                      href={resource.url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="plasmo-block plasmo-text-xs plasmo-font-medium plasmo-text-primary-700 hover:plasmo-underline"
                                    >
                                      {resource.title}
                                    </a>
                                    <div className="plasmo-mt-1 plasmo-text-[11px] plasmo-text-gray-500 plasmo-break-all">
                                      {resource.url}
                                    </div>
                                    <div className="plasmo-mt-2">
                                      <button
                                        onClick={() => void handleAddSuggestedResource(resource)}
                                        disabled={addingSuggestionUrl === resource.url}
                                        className="plasmo-rounded-md plasmo-bg-primary-600 plasmo-px-2 plasmo-py-1 plasmo-text-[11px] plasmo-font-semibold plasmo-text-white disabled:plasmo-opacity-50"
                                      >
                                        {addingSuggestionUrl === resource.url ? "Adding..." : "Add"}
                                      </button>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}

                            {getMatchedSuggestionResources(suggestion).length === 0 && (
                              <div className="plasmo-mt-2 plasmo-text-[11px] plasmo-text-gray-500">
                                No direct URL match found in Mind Reader suggestions for this tab yet.
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {msg.critiqueIssues && msg.critiqueIssues.length > 0 && (
                    <div className="plasmo-mt-3 plasmo-pt-3 plasmo-border-t plasmo-border-gray-200">
                      <div className="plasmo-mb-2 plasmo-flex plasmo-items-center plasmo-justify-between plasmo-gap-2">
                        <div className="plasmo-text-xs plasmo-font-semibold plasmo-text-gray-700">
                          👻 Ghost Critic Findings:
                        </div>
                        <div className="plasmo-flex plasmo-items-center plasmo-gap-2">
                          <button
                            type="button"
                            onClick={() => void highlightCritiqueOnPage(msg)}
                            disabled={!currentPage || highlightingCritiqueId === msg.id}
                            className="plasmo-rounded-md plasmo-border plasmo-border-rose-200 plasmo-bg-rose-50 plasmo-px-2.5 plasmo-py-1 plasmo-text-[11px] plasmo-font-semibold plasmo-text-rose-700 disabled:plasmo-cursor-not-allowed disabled:plasmo-opacity-50">
                            {highlightingCritiqueId === msg.id ? "Highlighting..." : "Highlight On Page"}
                          </button>
                          <button
                            type="button"
                            onClick={() => void clearCritiqueHighlights()}
                            disabled={!currentPage}
                            className="plasmo-rounded-md plasmo-border plasmo-border-slate-200 plasmo-bg-white plasmo-px-2.5 plasmo-py-1 plasmo-text-[11px] plasmo-font-semibold plasmo-text-slate-600 disabled:plasmo-cursor-not-allowed disabled:plasmo-opacity-50">
                            Clear
                          </button>
                        </div>
                      </div>
                      <div className="plasmo-space-y-2">
                        {msg.critiqueIssues.map((issue, idx) => {
                          const severityClassName =
                            issue.severity === "high"
                              ? "plasmo-bg-red-50 plasmo-border-red-200 plasmo-text-red-700"
                              : issue.severity === "medium"
                                ? "plasmo-bg-amber-50 plasmo-border-amber-200 plasmo-text-amber-700"
                                : "plasmo-bg-slate-50 plasmo-border-slate-200 plasmo-text-slate-700"

                          return (
                            <div
                              key={`${issue.title}-${idx}`}
                              className="plasmo-rounded-lg plasmo-border plasmo-border-slate-200 plasmo-bg-slate-50 plasmo-p-3"
                            >
                              <div className="plasmo-flex plasmo-items-start plasmo-justify-between plasmo-gap-2">
                                <div className="plasmo-min-w-0">
                                  <div className="plasmo-text-sm plasmo-font-semibold plasmo-text-slate-800">
                                    {issue.title}
                                  </div>
                                  <div className="plasmo-mt-1 plasmo-flex plasmo-flex-wrap plasmo-gap-1.5">
                                    <span
                                      className={`plasmo-rounded-full plasmo-border plasmo-px-2 plasmo-py-0.5 plasmo-text-[11px] plasmo-font-semibold ${severityClassName}`}
                                    >
                                      {issue.severity.toUpperCase()}
                                    </span>
                                    <span className="plasmo-rounded-full plasmo-border plasmo-border-slate-200 plasmo-bg-white plasmo-px-2 plasmo-py-0.5 plasmo-text-[11px] plasmo-font-medium plasmo-text-slate-600">
                                      {issue.category}
                                    </span>
                                    {issue.source_title && (
                                      <span className="plasmo-rounded-full plasmo-border plasmo-border-blue-200 plasmo-bg-blue-50 plasmo-px-2 plasmo-py-0.5 plasmo-text-[11px] plasmo-font-medium plasmo-text-blue-700">
                                        {issue.source_title}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>

                              <div className="plasmo-mt-2 plasmo-space-y-2">
                                <div>
                                  <div className="plasmo-text-[11px] plasmo-font-semibold plasmo-uppercase plasmo-tracking-[0.08em] plasmo-text-slate-500">
                                    Evidence
                                  </div>
                                  <div className="plasmo-mt-1 plasmo-text-xs plasmo-text-slate-700">
                                    {issue.evidence}
                                  </div>
                                </div>
                                <div>
                                  <div className="plasmo-text-[11px] plasmo-font-semibold plasmo-uppercase plasmo-tracking-[0.08em] plasmo-text-slate-500">
                                    Risk
                                  </div>
                                  <div className="plasmo-mt-1 plasmo-text-xs plasmo-text-slate-700">
                                    {issue.risk}
                                  </div>
                                </div>
                                <div>
                                  <div className="plasmo-text-[11px] plasmo-font-semibold plasmo-uppercase plasmo-tracking-[0.08em] plasmo-text-slate-500">
                                    Suggested Fix
                                  </div>
                                  <div className="plasmo-mt-1 plasmo-text-xs plasmo-text-slate-700">
                                    {issue.suggestion}
                                  </div>
                                </div>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </div>

                {msg.role === "assistant" && msg.diagnostics && (
                  (() => {
                    const { whyItems, technicalItems } = splitDiagnosticsForDisplay(msg.diagnostics)

                    return (
                      <div className="plasmo-mt-2 plasmo-space-y-2">
                        {whyItems.length > 0 && (
                          <details className="plasmo-rounded-lg plasmo-border plasmo-border-blue-100 plasmo-bg-blue-50/60 plasmo-px-3 plasmo-py-2">
                            <summary className="plasmo-cursor-pointer plasmo-text-[11px] plasmo-font-semibold plasmo-uppercase plasmo-tracking-[0.08em] plasmo-text-blue-700">
                              Why This Answer
                            </summary>
                            <div className="plasmo-mt-2 plasmo-space-y-2">
                              {whyItems.map((item) => (
                                <div key={item.label} className="plasmo-text-xs">
                                  <div className="plasmo-font-semibold plasmo-text-blue-900">
                                    {item.label}
                                  </div>
                                  <div className="plasmo-mt-0.5 plasmo-leading-relaxed plasmo-text-slate-700">
                                    {item.value}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </details>
                        )}

                        {technicalItems.length > 0 && (
                          <details className="plasmo-rounded-lg plasmo-border plasmo-border-gray-200 plasmo-bg-gray-50 plasmo-px-3 plasmo-py-2">
                            <summary className="plasmo-cursor-pointer plasmo-text-[11px] plasmo-font-semibold plasmo-uppercase plasmo-tracking-[0.08em] plasmo-text-gray-500">
                              Technical Diagnostics
                            </summary>
                            <div className="plasmo-mt-2 plasmo-grid plasmo-grid-cols-2 plasmo-gap-x-3 plasmo-gap-y-1">
                              {technicalItems.map((item) => (
                                <div key={item.label} className="plasmo-text-xs">
                                  <span className="plasmo-text-gray-500">{item.label}:</span>{" "}
                                  <span className="plasmo-font-medium plasmo-text-gray-700">{item.value}</span>
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    )
                  })()
                )}
              </div>
            </div>
          ))
        )}

        {isProcessing && (
          <div className="plasmo-flex plasmo-justify-start">
            <div className="plasmo-bg-white plasmo-border plasmo-border-gray-200 plasmo-rounded-lg plasmo-p-3">
              <div className="plasmo-flex plasmo-items-center plasmo-gap-2 plasmo-text-sm plasmo-text-gray-600">
                <div className="plasmo-animate-spin plasmo-h-4 plasmo-w-4 plasmo-border-2 plasmo-border-gray-300 plasmo-border-t-primary-600 plasmo-rounded-full" />
                <span>{processingStage || "Processing..."}</span>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer: Pinned Pages + Input */}
      <footer className="plasmo-border-t plasmo-border-gray-200 plasmo-bg-white">
        {/* Pinned Pages Strip */}
        {pinnedPages.length > 0 && (
          <div className="plasmo-border-b plasmo-border-gray-100 plasmo-p-2">
            <div className="plasmo-flex plasmo-items-center plasmo-justify-between plasmo-gap-3">
              <div className="plasmo-min-w-0">
                <div className="plasmo-text-[11px] plasmo-font-semibold plasmo-uppercase plasmo-tracking-[0.08em] plasmo-text-gray-500">
                  Context Basket
                </div>
                <div className={`plasmo-mt-1 plasmo-flex plasmo-items-center plasmo-gap-2 plasmo-text-xs ${
                  showTokenWarning ? "plasmo-text-orange-600 plasmo-font-semibold" : "plasmo-text-gray-600"
                }`}>
                  <span>
                    {pinnedPages.length} pages • ~{totalTokens.toLocaleString()} tokens
                  </span>
                  {showTokenWarning && (
                    <span className="plasmo-rounded-full plasmo-bg-orange-100 plasmo-px-2 plasmo-py-0.5 plasmo-text-[11px] plasmo-text-orange-800">
                      High token usage
                    </span>
                  )}
                </div>
              </div>
              <button
                onClick={() => setShowPinnedPagesList((prev) => !prev)}
                className="plasmo-flex-shrink-0 plasmo-rounded-md plasmo-border plasmo-border-gray-200 plasmo-bg-gray-50 plasmo-px-2.5 plasmo-py-1 plasmo-text-xs plasmo-font-medium plasmo-text-gray-700 hover:plasmo-bg-gray-100"
              >
                {showPinnedPagesList ? "Hide Pages" : "View Pages"}
              </button>
            </div>

            {showPinnedPagesList && (
              <div className="plasmo-mt-3">
                <input
                  type="text"
                  value={pinnedPageFilter}
                  onChange={(e) => setPinnedPageFilter(e.target.value)}
                  placeholder="Filter by title, URL, or domain"
                  className="plasmo-w-full plasmo-rounded-lg plasmo-border plasmo-border-gray-200 plasmo-bg-white plasmo-px-3 plasmo-py-2 plasmo-text-xs focus:plasmo-outline-none focus:plasmo-ring-2 focus:plasmo-ring-primary-500"
                />

                <div className="plasmo-mt-3 plasmo-max-h-52 plasmo-space-y-3 plasmo-overflow-y-auto plasmo-pr-1">
                  {groupedPinnedPages.length === 0 ? (
                    <div className="plasmo-rounded-lg plasmo-border plasmo-border-dashed plasmo-border-gray-200 plasmo-bg-gray-50 plasmo-p-3 plasmo-text-xs plasmo-text-gray-500">
                      No pinned pages match this filter.
                    </div>
                  ) : (
                    groupedPinnedPages.map(([hostname, pages]) => (
                      <div key={hostname} className="plasmo-space-y-2">
                        <div className="plasmo-sticky plasmo-top-0 plasmo-z-10 plasmo-inline-flex plasmo-items-center plasmo-gap-2 plasmo-rounded-full plasmo-bg-slate-100 plasmo-px-2.5 plasmo-py-1 plasmo-text-[11px] plasmo-font-semibold plasmo-text-slate-700">
                          <span>{hostname}</span>
                          <span className="plasmo-rounded-full plasmo-bg-white plasmo-px-1.5 plasmo-py-0.5 plasmo-text-[10px] plasmo-text-slate-500">
                            {pages.length}
                          </span>
                        </div>

                        {pages.map((page) => (
                          <div
                            key={page.id}
                            className="plasmo-flex plasmo-items-start plasmo-justify-between plasmo-gap-2 plasmo-rounded-lg plasmo-border plasmo-border-primary-100 plasmo-bg-primary-50/60 plasmo-p-2"
                          >
                            <div className="plasmo-min-w-0 plasmo-flex-1">
                              <div className="plasmo-flex plasmo-items-center plasmo-gap-2 plasmo-min-w-0">
                                {page.favicon && (
                                  <img src={page.favicon} alt="" className="plasmo-mt-0.5 plasmo-h-3.5 plasmo-w-3.5 plasmo-flex-shrink-0" />
                                )}
                                <div className="plasmo-min-w-0 plasmo-text-xs plasmo-font-medium plasmo-text-primary-900 plasmo-truncate">
                                  {page.title}
                                </div>
                              </div>
                              <div className="plasmo-mt-1 plasmo-text-[11px] plasmo-text-primary-700/80 plasmo-truncate">
                                {page.url} • ~{page.tokenEstimate.toLocaleString()} tokens
                              </div>
                            </div>
                            <button
                              onClick={() => unpinPage(page.id)}
                              className="plasmo-flex-shrink-0 plasmo-rounded-md plasmo-px-1.5 plasmo-py-0.5 plasmo-text-xs plasmo-font-semibold plasmo-text-primary-700 hover:plasmo-bg-primary-100 hover:plasmo-text-primary-900"
                              title="Unpin"
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Input Composer */}
        <div className="plasmo-p-3">
          <div className="plasmo-flex plasmo-gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                pinnedPages.length === 0
                  ? "Pin pages first..."
                  : "Ask a question, request a diagram, or ask for a critique/review"
              }
              disabled={pinnedPages.length === 0 || isProcessing}
              className="plasmo-flex-1 plasmo-px-3 plasmo-py-2 plasmo-text-sm plasmo-border plasmo-border-gray-300 plasmo-rounded-lg focus:plasmo-outline-none focus:plasmo-ring-2 focus:plasmo-ring-primary-500 disabled:plasmo-bg-gray-100 disabled:plasmo-cursor-not-allowed"
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || pinnedPages.length === 0 || isProcessing}
              className="plasmo-px-4 plasmo-py-2 plasmo-bg-primary-600 plasmo-text-white plasmo-rounded-lg plasmo-text-sm plasmo-font-medium hover:plasmo-bg-primary-700 plasmo-transition-colors disabled:plasmo-opacity-50 disabled:plasmo-cursor-not-allowed"
            >
              Send
            </button>
          </div>
          {pinnedPages.length > 0 && (
            <div className="plasmo-mt-2 plasmo-text-xs plasmo-text-gray-500">
              Tip: ask for a flow/sequence diagram to trigger Visual Architect, or say review/critique to trigger Ghost Critic.
            </div>
          )}
        </div>
      </footer>
    </div>
  )
}

export default SidePanel
