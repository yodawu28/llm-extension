import type { PlasmoCSConfig } from "plasmo"

import { EXTENSION_BUILD_LABEL } from "~lib/build-info"
import { extractScrapedContent, type ScrapedContent } from "~lib/page-content-extractor"

export const config: PlasmoCSConfig = {
  matches: ["<all_urls>"],
  all_frames: false,
  run_at: "document_end"
}

interface MindReaderResource {
  url: string
  title: string
  type: "confluence" | "jira" | "github" | "doc" | "generic"
  context: string
  confidence: "high" | "medium" | "low"
}

interface MindReaderStatusPayload {
  tone: "loading" | "success" | "error"
  title: string
  message: string
  autoDismissMs?: number
  progress?: {
    completed: number
    total: number
    current?: string
  }
}

interface GhostCriticIssue {
  title: string
  severity: "high" | "medium" | "low"
  category: string
  evidence: string
  risk?: string
  suggestion?: string
  source_title?: string
}

const TOAST_ROOT_ID = "__mind-reader-toast"
const GHOST_CRITIC_TOAST_ROOT_ID = "__ghost-critic-toast"
const GHOST_CRITIC_STYLE_ID = "__ghost-critic-style"
const GHOST_CRITIC_HIGHLIGHT_NAMES = {
  high: "ghost-critic-high",
  medium: "ghost-critic-medium",
  low: "ghost-critic-low"
} as const
const GHOST_CRITIC_STOP_WORDS = new Set([
  "about",
  "after",
  "before",
  "could",
  "from",
  "have",
  "into",
  "that",
  "there",
  "these",
  "this",
  "those",
  "when",
  "where",
  "which",
  "while",
  "with",
  "without",
  "would"
])
const MAX_DIRECT_IMPORT_PAGES = 12
let toastDismissTimer: number | null = null
let ghostCriticToastDismissTimer: number | null = null
let currentImportId: string | null = null

function clearToastTimer() {
  if (toastDismissTimer != null) {
    window.clearTimeout(toastDismissTimer)
    toastDismissTimer = null
  }
}

function removeMindReaderToast(notifyBackground = false) {
  clearToastTimer()
  document.getElementById(TOAST_ROOT_ID)?.remove()

  if (notifyBackground) {
    void chrome.runtime.sendMessage({ action: "mind-reader-dismiss" }).catch(() => {})
  }
}

function clearGhostCriticToastTimer() {
  if (ghostCriticToastDismissTimer != null) {
    window.clearTimeout(ghostCriticToastDismissTimer)
    ghostCriticToastDismissTimer = null
  }
}

function removeGhostCriticToast() {
  clearGhostCriticToastTimer()
  document.getElementById(GHOST_CRITIC_TOAST_ROOT_ID)?.remove()
}

function createToastButton(
  label: string,
  variant: "primary" | "secondary" | "ghost",
  onClick: () => void | Promise<void>
) {
  const button = document.createElement("button")
  button.type = "button"
  button.textContent = label

  Object.assign(button.style, {
    border: variant === "ghost" ? "1px solid transparent" : "1px solid transparent",
    borderRadius: "10px",
    padding: "10px 14px",
    fontSize: "13px",
    fontWeight: "600",
    cursor: "pointer",
    transition: "all 120ms ease",
    outline: "none"
  })

  if (variant === "primary") {
    button.style.background = "#2563eb"
    button.style.color = "#ffffff"
    button.style.boxShadow = "0 8px 18px rgba(37, 99, 235, 0.22)"
  } else if (variant === "secondary") {
    button.style.background = "#eff6ff"
    button.style.color = "#1d4ed8"
    button.style.border = "1px solid #bfdbfe"
  } else {
    button.style.background = "transparent"
    button.style.color = "#6b7280"
  }

  button.addEventListener("click", () => {
    void onClick()
  })

  return button
}

function createToastShell(accentColor: string, rootId: string = TOAST_ROOT_ID) {
  if (rootId === TOAST_ROOT_ID) {
    removeMindReaderToast(false)
  } else {
    removeGhostCriticToast()
  }

  const root = document.createElement("div")
  root.id = rootId

  Object.assign(root.style, {
    position: "fixed",
    top: "20px",
    right: "20px",
    width: "min(420px, calc(100vw - 32px))",
    zIndex: "2147483647",
    background: "#ffffff",
    color: "#111827",
    border: `1px solid ${accentColor}`,
    borderRadius: "16px",
    boxShadow: "0 24px 60px rgba(15, 23, 42, 0.25)",
    overflow: "hidden",
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
  })

  const header = document.createElement("div")
  Object.assign(header.style, {
    height: "4px",
    background: accentColor
  })

  const body = document.createElement("div")
  Object.assign(body.style, {
    padding: "16px"
  })

  const footer = document.createElement("div")
  footer.textContent = EXTENSION_BUILD_LABEL
  Object.assign(footer.style, {
    padding: "0 16px 12px",
    fontSize: "10px",
    fontWeight: "700",
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "#64748b"
  })

  root.append(header, body, footer)
  document.body.appendChild(root)

  return { root, body, footer }
}

function ensureGhostCriticStyles() {
  if (document.getElementById(GHOST_CRITIC_STYLE_ID)) {
    return
  }

  const style = document.createElement("style")
  style.id = GHOST_CRITIC_STYLE_ID
  style.textContent = `
    ::highlight(${GHOST_CRITIC_HIGHLIGHT_NAMES.high}) {
      background: rgba(248, 113, 113, 0.28);
      color: inherit;
    }

    ::highlight(${GHOST_CRITIC_HIGHLIGHT_NAMES.medium}) {
      background: rgba(251, 191, 36, 0.30);
      color: inherit;
    }

    ::highlight(${GHOST_CRITIC_HIGHLIGHT_NAMES.low}) {
      background: rgba(125, 211, 252, 0.26);
      color: inherit;
    }
  `

  document.documentElement.appendChild(style)
}

function getGhostCriticHighlightSupport() {
  const cssHighlights = (globalThis.CSS as any)?.highlights
  const HighlightCtor = (globalThis as any).Highlight

  if (!cssHighlights || typeof HighlightCtor !== "function") {
    return null
  }

  return {
    cssHighlights: cssHighlights as Map<string, any>,
    HighlightCtor: HighlightCtor as new (...ranges: Range[]) => any
  }
}

function clearGhostCriticHighlights() {
  const support = getGhostCriticHighlightSupport()

  if (!support) {
    return false
  }

  Object.values(GHOST_CRITIC_HIGHLIGHT_NAMES).forEach((name) => {
    support.cssHighlights.delete(name)
  })

  removeGhostCriticToast()
  return true
}

function normalizeSearchText(text: string) {
  return text.replace(/\s+/g, " ").replace(/[“”‘’`]/g, "\"").trim()
}

function tokenizeGhostCriticText(text: string): string[] {
  return normalizeSearchText(text)
    .toLowerCase()
    .split(/[^a-z0-9]+/i)
    .map((token) => token.trim())
    .filter((token) => token.length >= 4 && !GHOST_CRITIC_STOP_WORDS.has(token))
}

function buildCollapsedIndex(text: string) {
  let normalized = ""
  const indexMap: number[] = []
  let lastWasWhitespace = true

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]
    const isWhitespace = /\s/.test(char)

    if (isWhitespace) {
      if (lastWasWhitespace) {
        continue
      }

      normalized += " "
      indexMap.push(index)
      lastWasWhitespace = true
      continue
    }

    normalized += char.toLowerCase()
    indexMap.push(index)
    lastWasWhitespace = false
  }

  if (normalized.endsWith(" ")) {
    normalized = normalized.slice(0, -1)
    indexMap.pop()
  }

  return {
    normalized,
    indexMap
  }
}

function buildGhostCriticSearchCorpus(issue: GhostCriticIssue) {
  return [
    issue.title,
    issue.evidence,
    issue.risk || "",
    issue.suggestion || ""
  ]
    .map((part) => normalizeSearchText(part))
    .filter(Boolean)
    .join(". ")
}

function buildGhostCriticSearchCandidates(issue: GhostCriticIssue): string[] {
  const normalizedEvidence = normalizeSearchText(
    issue.evidence
      .replace(/^[\s"'`-]+|[\s"'`-]+$/g, "")
      .replace(/\b(paraphrased|paraphrase|approx\.?|approximately)\b:?/gi, "")
  )
  const normalizedTitle = normalizeSearchText(issue.title)
  const normalizedRisk = normalizeSearchText(issue.risk || "")
  const normalizedSuggestion = normalizeSearchText(issue.suggestion || "")

  const candidates = new Set<string>()

  const quotedMatches = Array.from(
    issue.evidence.matchAll(/["“](.+?)["”]/g),
    (match) => normalizeSearchText(match[1] || "")
  )

  quotedMatches.forEach((candidate) => {
    if (candidate.length >= 18) {
      candidates.add(candidate)
    }
  })

  if (normalizedEvidence.length >= 18) {
    candidates.add(normalizedEvidence)
  }

  if (normalizedTitle.length >= 12) {
    candidates.add(normalizedTitle)
  }

  if (normalizedTitle && normalizedEvidence) {
    candidates.add(`${normalizedTitle}. ${normalizedEvidence}`.slice(0, 220))
  }

  normalizedEvidence
    .split(/(?:\.\s+|\n+|•)/)
    .map((part) => normalizeSearchText(part))
    .filter((part) => part.length >= 24)
    .forEach((part) => candidates.add(part))

  if (normalizedRisk.length >= 24) {
    candidates.add(normalizedRisk)
  }

  if (normalizedSuggestion.length >= 24) {
    candidates.add(normalizedSuggestion)
  }

  const words = normalizedEvidence.split(/\s+/).filter(Boolean)

  if (words.length >= 8) {
    candidates.add(words.slice(0, Math.min(words.length, 12)).join(" "))
  }

  if (words.length >= 14) {
    candidates.add(words.slice(0, Math.min(words.length, 18)).join(" "))
  }

  const titleWords = normalizedTitle.split(/\s+/).filter(Boolean)
  if (titleWords.length >= 3) {
    candidates.add(titleWords.slice(0, Math.min(titleWords.length, 8)).join(" "))
  }

  return Array.from(candidates).sort((left, right) => right.length - left.length)
}

function shouldSkipTextNode(node: Node | null) {
  if (!node || node.nodeType !== Node.TEXT_NODE) {
    return true
  }

  const text = node.textContent || ""
  if (!text.trim()) {
    return true
  }

  const parent = node.parentElement
  if (!parent) {
    return true
  }

  if (parent.closest(`#${TOAST_ROOT_ID}, #${GHOST_CRITIC_TOAST_ROOT_ID}`)) {
    return true
  }

  const tagName = parent.tagName.toLowerCase()
  return [
    "script",
    "style",
    "noscript",
    "textarea",
    "input",
    "select",
    "option",
    "button"
  ].includes(tagName)
}

function shouldSkipGhostCriticElement(element: HTMLElement | null) {
  if (!element) {
    return true
  }

  if (element.closest(`#${TOAST_ROOT_ID}, #${GHOST_CRITIC_TOAST_ROOT_ID}`)) {
    return true
  }

  const tagName = element.tagName.toLowerCase()
  if (
    [
      "body",
      "html",
      "script",
      "style",
      "noscript",
      "textarea",
      "input",
      "select",
      "option",
      "button",
      "svg",
      "path"
    ].includes(tagName)
  ) {
    return true
  }

  const text = normalizeSearchText(element.innerText || element.textContent || "")
  return text.length < 24 || text.length > 900
}

function findFuzzyBlockRange(issue: GhostCriticIssue, usedElements: Set<HTMLElement>): Range | null {
  const evidenceTokens = Array.from(new Set(tokenizeGhostCriticText(buildGhostCriticSearchCorpus(issue))))

  if (evidenceTokens.length < 3) {
    return null
  }

  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT)
  let bestElement: HTMLElement | null = null
  let bestScore = 0

  while (walker.nextNode()) {
    const element = walker.currentNode as HTMLElement

    if (usedElements.has(element) || shouldSkipGhostCriticElement(element)) {
      continue
    }

    const text = normalizeSearchText(element.innerText || element.textContent || "")
    const elementTokens = new Set(tokenizeGhostCriticText(text))

    if (elementTokens.size < 4) {
      continue
    }

    const matchedTokenCount = evidenceTokens.filter((token) => elementTokens.has(token)).length
    const minRequiredMatches = evidenceTokens.length >= 10 ? 4 : 2

    if (matchedTokenCount < minRequiredMatches) {
      continue
    }

    const overlapScore = matchedTokenCount / evidenceTokens.length
    const densityPenalty = Math.max(0, text.length - 320) / 2000
    const score = overlapScore + matchedTokenCount * 0.03 - densityPenalty

    if (score <= bestScore) {
      continue
    }

    bestScore = score
    bestElement = element
  }

  if (!bestElement || bestScore < 0.24) {
    return null
  }

  usedElements.add(bestElement)

  const range = document.createRange()
  range.selectNodeContents(bestElement)
  return range
}

function findMatchingRange(candidate: string): Range | null {
  const normalizedCandidate = buildCollapsedIndex(candidate).normalized

  if (!normalizedCandidate || normalizedCandidate.length < 18) {
    return null
  }

  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)

  while (walker.nextNode()) {
    const node = walker.currentNode

    if (shouldSkipTextNode(node)) {
      continue
    }

    const text = node.textContent || ""
    const collapsed = buildCollapsedIndex(text)
    const matchIndex = collapsed.normalized.indexOf(normalizedCandidate)

    if (matchIndex === -1) {
      continue
    }

    const startIndex = collapsed.indexMap[matchIndex]
    const endMapIndex = matchIndex + normalizedCandidate.length
    const endIndex = collapsed.indexMap[endMapIndex] ?? text.length

    if (startIndex == null || endIndex <= startIndex) {
      continue
    }

    const range = document.createRange()
    range.setStart(node, startIndex)
    range.setEnd(node, endIndex)
    return range
  }

  return null
}

function renderGhostCriticToast(matchedIssues: number, totalIssues: number) {
  const { body } = createToastShell("#dc2626", GHOST_CRITIC_TOAST_ROOT_ID)

  const title = document.createElement("div")
  title.textContent =
    matchedIssues > 0
      ? `Ghost Critic highlighted ${matchedIssues} finding${matchedIssues > 1 ? "s" : ""}`
      : "Ghost Critic could not find matching text on this page"
  Object.assign(title.style, {
    fontSize: "15px",
    fontWeight: "700",
    color: "#111827"
  })

  const message = document.createElement("div")
  message.textContent =
    matchedIssues > 0
      ? `${matchedIssues} of ${totalIssues} issue${totalIssues > 1 ? "s" : ""} matched the current page content.`
      : "Try opening the source document that contains the evidence, then run highlight again."
  Object.assign(message.style, {
    marginTop: "6px",
    fontSize: "13px",
    lineHeight: "1.5",
    color: "#4b5563"
  })

  const actions = document.createElement("div")
  Object.assign(actions.style, {
    display: "flex",
    gap: "8px",
    marginTop: "14px"
  })

  const clearButton = createToastButton("Clear Highlights", "secondary", () => {
    clearGhostCriticHighlights()
  })

  const closeButton = createToastButton("Close", "ghost", () => {
    removeGhostCriticToast()
  })

  actions.append(clearButton, closeButton)
  body.append(title, message, actions)

  clearGhostCriticToastTimer()
  ghostCriticToastDismissTimer = window.setTimeout(() => {
    removeGhostCriticToast()
  }, matchedIssues > 0 ? 5000 : 6500)
}

function highlightGhostCriticIssues(issues: GhostCriticIssue[]) {
  const support = getGhostCriticHighlightSupport()

  if (!support) {
    return {
      success: false,
      error: "CSS Highlights API is not available on this page."
    }
  }

  ensureGhostCriticStyles()
  clearGhostCriticHighlights()

  const rangesBySeverity: Record<GhostCriticIssue["severity"], Range[]> = {
    high: [],
    medium: [],
    low: []
  }
  let matchedIssues = 0
  let firstMatchElement: HTMLElement | null = null
  const usedFuzzyElements = new Set<HTMLElement>()

  issues.forEach((issue) => {
    const candidates = buildGhostCriticSearchCandidates(issue)
    let matchedRange: Range | null = null

    for (const candidate of candidates) {
      const range = findMatchingRange(candidate)

      if (!range) {
        continue
      }

      matchedRange = range
      break
    }

    if (!matchedRange) {
      matchedRange = findFuzzyBlockRange(issue, usedFuzzyElements)
    }

    if (!matchedRange) {
      return
    }

    rangesBySeverity[issue.severity].push(matchedRange)
    matchedIssues += 1

    if (!firstMatchElement && matchedRange.startContainer.parentElement instanceof HTMLElement) {
      firstMatchElement = matchedRange.startContainer.parentElement
    }
  })

  Object.entries(rangesBySeverity).forEach(([severity, ranges]) => {
    if (ranges.length === 0) {
      return
    }

    const highlightName =
      GHOST_CRITIC_HIGHLIGHT_NAMES[severity as keyof typeof GHOST_CRITIC_HIGHLIGHT_NAMES]
    support.cssHighlights.set(highlightName, new support.HighlightCtor(...ranges))
  })

  const firstMatchTarget = firstMatchElement as HTMLElement | null

  if (firstMatchTarget) {
    firstMatchTarget.scrollIntoView({
      behavior: "smooth",
      block: "center"
    })
  }

  renderGhostCriticToast(matchedIssues, issues.length)

  return {
    success: true,
    matchedIssues,
    totalIssues: issues.length
  }
}

function renderStatusToast(
  tone: "loading" | "success" | "error",
  title: string,
  message: string,
  autoDismissMs: number = 0,
  progress?: MindReaderStatusPayload["progress"]
) {
  const accentColor =
    tone === "success" ? "#16a34a" : tone === "error" ? "#dc2626" : "#2563eb"
  const icon = tone === "success" ? "✓" : tone === "error" ? "!" : "⋯"

  const { body } = createToastShell(accentColor)

  const row = document.createElement("div")
  Object.assign(row.style, {
    display: "flex",
    gap: "12px",
    alignItems: "flex-start"
  })

  const iconEl = document.createElement("div")
  iconEl.textContent = icon
  Object.assign(iconEl.style, {
    width: "28px",
    height: "28px",
    borderRadius: "999px",
    background: `${accentColor}16`,
    color: accentColor,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "700",
    flexShrink: "0"
  })

  const content = document.createElement("div")
  content.style.flex = "1"

  const titleEl = document.createElement("div")
  titleEl.textContent = title
  Object.assign(titleEl.style, {
    fontSize: "15px",
    fontWeight: "700"
  })

  const messageEl = document.createElement("div")
  messageEl.textContent = message
  Object.assign(messageEl.style, {
    marginTop: "6px",
    fontSize: "13px",
    lineHeight: "1.5",
    color: "#4b5563"
  })

  const progressContainer = document.createElement("div")

  if (progress && progress.total > 0) {
    Object.assign(progressContainer.style, {
      marginTop: "12px"
    })

    const progressMeta = document.createElement("div")
    progressMeta.textContent = `${progress.completed} / ${progress.total}${progress.current ? ` • ${progress.current}` : ""}`
    Object.assign(progressMeta.style, {
      fontSize: "12px",
      color: "#6b7280",
      marginBottom: "6px"
    })

    const track = document.createElement("div")
    Object.assign(track.style, {
      width: "100%",
      height: "8px",
      borderRadius: "999px",
      background: "#e5e7eb",
      overflow: "hidden"
    })

    const fill = document.createElement("div")
    Object.assign(fill.style, {
      width: `${Math.max(0, Math.min(100, (progress.completed / progress.total) * 100))}%`,
      height: "100%",
      borderRadius: "999px",
      background: accentColor,
      transition: "width 180ms ease"
    })

    track.append(fill)
    progressContainer.append(progressMeta, track)
  }

  const actionRow = document.createElement("div")
  Object.assign(actionRow.style, {
    display: "flex",
    gap: "8px",
    marginTop: "14px"
  })

  if (tone === "loading" && currentImportId) {
    const cancelButton = createToastButton("Cancel Import", "secondary", async () => {
      await handleCancelImportFromToast()
    })
    actionRow.append(cancelButton)
  }

  const dismissButton = createToastButton("Close", "ghost", () => {
    removeMindReaderToast(false)
  })
  actionRow.append(dismissButton)

  content.append(titleEl, messageEl)
  if (progressContainer.childNodes.length > 0) {
    content.append(progressContainer)
  }
  content.append(actionRow)
  row.append(iconEl, content)
  body.append(row)

  if (autoDismissMs > 0) {
    clearToastTimer()
    toastDismissTimer = window.setTimeout(() => {
      removeMindReaderToast(false)
    }, autoDismissMs)
  }
}

async function handleImportSelectedFromToast(resources: MindReaderResource[]) {
  currentImportId = crypto.randomUUID()

  renderStatusToast(
    "loading",
    "Importing related pages",
    `Fetching ${resources.length} selected page${resources.length > 1 ? "s" : ""}...`,
    0,
    {
      completed: 0,
      total: resources.length
    }
  )

  try {
    const response = await chrome.runtime.sendMessage({
      action: "mind-reader-add-all",
      payload: {
        resources,
        importId: currentImportId
      }
    })

    if (!response?.success) {
      throw new Error(response?.error || "Failed to add resources")
    }
  } catch (error) {
    console.error("[Mind Reader] Failed to add resources from toast:", error)
    renderStatusToast(
      "error",
      "Could not add pages",
      "Fetching related documents failed. Try again from the panel.",
      5000
    )
  } finally {
    currentImportId = null
  }
}

async function handleCancelImportFromToast() {
  if (!currentImportId) return

  const importId = currentImportId
  currentImportId = null

  await chrome.runtime.sendMessage({
    action: "mind-reader-cancel-import",
    payload: { importId }
  })

  renderStatusToast(
    "error",
    "Cancelling import",
    "Stopping the remaining page imports...",
    1500
  )
}

function showMindReaderToast(resources: MindReaderResource[]) {
  const { body } = createToastShell("#2563eb")
  const selectedUrls = new Set(resources.slice(0, MAX_DIRECT_IMPORT_PAGES).map((resource) => resource.url))

  const titleRow = document.createElement("div")
  Object.assign(titleRow.style, {
    display: "flex",
    gap: "12px",
    alignItems: "flex-start"
  })

  const iconEl = document.createElement("div")
  iconEl.textContent = "💡"
  Object.assign(iconEl.style, {
    fontSize: "20px",
    lineHeight: "1",
    flexShrink: "0"
  })

  const content = document.createElement("div")
  content.style.flex = "1"

  const titleEl = document.createElement("div")
  titleEl.textContent = `Mind Reader found ${resources.length} related document${resources.length > 1 ? "s" : ""}`
  Object.assign(titleEl.style, {
    fontSize: "15px",
    fontWeight: "700"
  })

  const subtitleEl = document.createElement("div")
  subtitleEl.textContent = "Pick the pages you want to import directly into context."
  Object.assign(subtitleEl.style, {
    marginTop: "6px",
    fontSize: "13px",
    lineHeight: "1.5",
    color: "#4b5563"
  })

  content.append(titleEl, subtitleEl)
  titleRow.append(iconEl, content)
  body.append(titleRow)

  const list = document.createElement("div")
  Object.assign(list.style, {
    marginTop: "14px",
    display: "grid",
    gap: "8px",
    maxHeight: "280px",
    overflowY: "auto",
    paddingRight: "4px"
  })

  const helper = document.createElement("div")
  Object.assign(helper.style, {
    marginTop: "12px",
    fontSize: "12px",
    color: "#6b7280"
  })

  const updateHelperText = () => {
    helper.textContent = `Selected ${selectedUrls.size} / ${Math.min(resources.length, MAX_DIRECT_IMPORT_PAGES)} pages for direct import`
  }

  resources.forEach((resource) => {
    const item = document.createElement("label")
    Object.assign(item.style, {
      border: "1px solid #e5e7eb",
      borderRadius: "12px",
      padding: "10px 12px",
      background: "#f8fafc",
      display: "flex",
      gap: "10px",
      alignItems: "flex-start",
      cursor: "pointer"
    })

    const checkbox = document.createElement("input")
    checkbox.type = "checkbox"
    checkbox.checked = selectedUrls.has(resource.url)
    checkbox.style.marginTop = "2px"
    checkbox.style.flexShrink = "0"

    const resourceTitle = document.createElement("div")
    resourceTitle.textContent = resource.title || resource.url
    Object.assign(resourceTitle.style, {
      fontSize: "13px",
      fontWeight: "600",
      color: "#111827"
    })

    const meta = document.createElement("div")
    meta.textContent = `${resource.type} • ${resource.confidence}`
    Object.assign(meta.style, {
      marginTop: "4px",
      fontSize: "12px",
      color: "#6b7280"
    })

    const textWrap = document.createElement("div")
    textWrap.style.flex = "1"
    textWrap.append(resourceTitle, meta)

    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        if (selectedUrls.size >= MAX_DIRECT_IMPORT_PAGES) {
          checkbox.checked = false
          helper.textContent = `You can import up to ${MAX_DIRECT_IMPORT_PAGES} pages at once.`
          return
        }
        selectedUrls.add(resource.url)
      } else {
        selectedUrls.delete(resource.url)
      }

      updateHelperText()
    })

    item.append(checkbox, textWrap)
    list.append(item)
  })

  body.append(list)
  body.append(helper)
  updateHelperText()

  if (resources.length > MAX_DIRECT_IMPORT_PAGES) {
    const limitNote = document.createElement("div")
    limitNote.textContent = `To avoid overload, direct import is limited to ${MAX_DIRECT_IMPORT_PAGES} pages per run.`
    Object.assign(limitNote.style, {
      marginTop: "8px",
      fontSize: "12px",
      color: "#92400e"
    })
    body.append(limitNote)
  }

  const actions = document.createElement("div")
  Object.assign(actions.style, {
    display: "flex",
    gap: "8px",
    marginTop: "16px"
  })

  const importSelectedButton = createToastButton("Import Selected", "primary", async () => {
    const selectedResources = resources.filter((resource) => selectedUrls.has(resource.url))

    if (selectedResources.length === 0) {
      helper.textContent = "Select at least one page to import."
      return
    }

    await handleImportSelectedFromToast(selectedResources)
  })

  const openPanelButton = createToastButton("Open Panel", "secondary", async () => {
    await chrome.runtime.sendMessage({ action: "mind-reader-open-panel" })
    removeMindReaderToast(false)
  })

  const dismissButton = createToastButton("Dismiss", "ghost", () => {
    removeMindReaderToast(true)
  })

  actions.append(importSelectedButton, openPanelButton, dismissButton)
  body.append(actions)
}

/**
 * Main scraping function
 */
export function scrapePage(): ScrapedContent | null {
  const scraped = extractScrapedContent(document, window.location.href, document.title.trim())

  if (!scraped) {
    console.warn("[Page Scraper] No content found on this page")
    return null
  }

  console.log(
    `[Page Scraper] Scraped: ${scraped.title} (~${scraped.metadata.tokenEstimate} tokens, ${scraped.metadata.wordCount} words)`
  )

  return scraped
}

// Listen for messages from sidepanel and background
chrome.runtime.onMessage.addListener((message: any, _sender: any, sendResponse: any) => {
  if (message.action === "scrape-page") {
    const result = scrapePage()
    sendResponse({ success: !!result, data: result })
  } else if (message.action === "scan-links") {
    // Mind Reader: Scan for related links
    // Note: Cannot use dynamic import in content script, so scanning happens in background
    sendResponse({
      success: false,
      error: "Link scanning should be done in background service worker"
    })
  } else if (message.action === "show-mind-reader-toast") {
    const resources: MindReaderResource[] = message.payload?.resources || []
    showMindReaderToast(resources)
    sendResponse({ success: true })
  } else if (message.action === "show-mind-reader-status") {
    const payload: MindReaderStatusPayload = message.payload
    if (payload?.tone !== "loading") {
      currentImportId = null
    }
    renderStatusToast(
      payload?.tone || "success",
      payload?.title || "Mind Reader",
      payload?.message || "",
      payload?.autoDismissMs || 4000,
      payload?.progress
    )
    sendResponse({ success: true })
  } else if (message.action === "ghost-critic-highlight") {
    const issues: GhostCriticIssue[] = Array.isArray(message.payload?.issues)
      ? message.payload.issues
      : []

    sendResponse(highlightGhostCriticIssues(issues))
  } else if (message.action === "ghost-critic-clear-highlights") {
    sendResponse({
      success: clearGhostCriticHighlights()
    })
  }
  return true // Keep message channel open for async response
})

console.log("[Page Scraper] Content script loaded for all pages")
