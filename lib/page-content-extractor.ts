import DOMPurify from "dompurify"

import { estimateTokenCount, htmlToMarkdown } from "~lib/markdown-converter"

export type ScrapedPageType = "confluence" | "jira" | "generic"

export interface ScrapedContent {
  title: string
  url: string
  htmlContent: string
  markdownContent: string
  metadata: {
    type: ScrapedPageType
    tokenEstimate: number
    wordCount: number
  }
}

export function detectPageType(url: string): ScrapedPageType {
  if (
    url.includes("atlassian.net/wiki") ||
    url.includes("confluence.com") ||
    url.includes("/pages/viewpage.action") ||
    /wiki\./i.test(url)
  ) {
    return "confluence"
  }

  if (
    url.includes("atlassian.net/browse") ||
    url.includes("jira.com") ||
    /jira\./i.test(url) ||
    /\/browse\/[A-Z]+-\d+/i.test(url)
  ) {
    return "jira"
  }

  return "generic"
}

function findMainContent(doc: Document, pageType: ScrapedPageType): HTMLElement | null {
  if (pageType === "confluence") {
    const confluenceSelectors = [
      "#main-content .wiki-content",
      "[data-testid='content-body']",
      ".page-content .wiki-content",
      "#content .content-body"
    ]

    for (const selector of confluenceSelectors) {
      const element = doc.querySelector(selector)
      if (element) return element as HTMLElement
    }
  }

  if (pageType === "jira") {
    const jiraSelectors = [
      "[data-testid='issue.views.issue-base.foundation.summary']",
      ".issue-view",
      "[data-testid='issue-view-foundation.ui.issue-view-base.issue-view-container']"
    ]

    for (const selector of jiraSelectors) {
      const element = doc.querySelector(selector)
      if (element) return element as HTMLElement
    }
  }

  const genericSelectors = [
    "main",
    "article",
    "[role='main']",
    ".main-content",
    "#main-content",
    "#content",
    ".content",
    ".post-content",
    ".article-content"
  ]

  for (const selector of genericSelectors) {
    const element = doc.querySelector(selector)
    if (element && element.textContent && element.textContent.trim().length > 100) {
      return element as HTMLElement
    }
  }

  if (!doc.body) {
    return null
  }

  const body = doc.body.cloneNode(true) as HTMLElement
  body.querySelectorAll("nav, header, footer, aside, script, style").forEach((el) => el.remove())
  return body
}

function cleanHTML(html: string, pageType: ScrapedPageType): string {
  const parser = new DOMParser()
  const doc = parser.parseFromString(html, "text/html")

  const noiseSelectors = [
    "script",
    "style",
    "iframe",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    ".advertisement",
    ".ads",
    ".sidebar",
    ".comments",
    ".related-posts",
    "[role='navigation']",
    "[role='banner']",
    "[role='contentinfo']"
  ]

  if (pageType === "confluence") {
    noiseSelectors.push(
      ".confluence-information-macro",
      ".confluence-macro",
      "[data-macro-name]",
      ".expand-control",
      ".status-macro",
      ".table-of-contents",
      ".aui-badge",
      ".aui-lozenge",
      ".aui-icon",
      ".breadcrumbs",
      ".page-metadata",
      ".page-metadata-banner",
      ".content-toolbar",
      ".page-toolbar",
      ".editor-toolbar",
      ".code-block-toolbar",
      ".copy-button",
      ".like-button",
      ".comment-count",
      ".view-count",
      ".page-layout-sidebar",
      ".ia-fixed-sidebar",
      "#navigation",
      ".space-navigation",
      ".confluence-embedded-file-wrapper",
      ".attachment-preview",
      ".embedded-file",
      ".version-history",
      ".page-contributors",
      ".created-date",
      ".modified-date",
      "[data-confluence-connect]",
      ".ap-container",
      ".edit-mode",
      "[contenteditable='true']"
    )
  }

  if (pageType === "jira") {
    noiseSelectors.push(
      "[data-testid*='navigation']",
      ".issue-navigator",
      ".navigator-sidebar",
      ".issue-actions",
      ".issue-toolbar",
      ".operations-container",
      ".action-buttons",
      ".comment-input",
      ".edit-issue",
      ".attachment-thumbnails",
      ".issue-links-container",
      ".people-details",
      ".activity-panel",
      ".issue-header-content",
      ".people-area",
      ".dates-area",
      ".votes-area",
      ".watchers-area",
      ".jira-connect-addon",
      "[data-jira-connect]"
    )
  }

  noiseSelectors.forEach((selector) => {
    doc.querySelectorAll(selector).forEach((el) => el.remove())
  })

  doc.querySelectorAll("p:empty, div:empty, span:empty").forEach((el) => el.remove())

  return doc.body.innerHTML
}

export function extractScrapedContent(
  doc: Document,
  url: string,
  title: string = doc.title.trim()
): ScrapedContent | null {
  const pageType = detectPageType(url)
  const contentElement = findMainContent(doc, pageType)

  if (!contentElement) {
    return null
  }

  const rawHTML = contentElement.innerHTML
  const cleanedHTML = cleanHTML(rawHTML, pageType)

  const sanitizedHTML = DOMPurify.sanitize(cleanedHTML, {
    ALLOWED_TAGS: [
      "p",
      "br",
      "strong",
      "em",
      "u",
      "a",
      "ul",
      "ol",
      "li",
      "h1",
      "h2",
      "h3",
      "h4",
      "h5",
      "h6",
      "blockquote",
      "code",
      "pre",
      "table",
      "thead",
      "tbody",
      "tr",
      "th",
      "td",
      "img"
    ],
    ALLOWED_ATTR: ["href", "title", "data-language", "src", "alt"]
  })

  const markdown = htmlToMarkdown(sanitizedHTML, url)
  const tokenEstimate = estimateTokenCount(markdown)
  const wordCount = markdown.split(/\s+/).filter(Boolean).length

  return {
    title,
    url,
    htmlContent: sanitizedHTML,
    markdownContent: markdown,
    metadata: {
      type: pageType,
      tokenEstimate,
      wordCount
    }
  }
}

export function extractScrapedContentFromHTML(
  html: string,
  url: string,
  fallbackTitle: string = ""
): ScrapedContent | null {
  const parser = new DOMParser()
  const doc = parser.parseFromString(html, "text/html")
  const title = doc.title.trim() || fallbackTitle || new URL(url).hostname

  return extractScrapedContent(doc, url, title)
}
