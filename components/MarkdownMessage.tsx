import React from "react"

import { MermaidDiagram } from "~components/MermaidDiagram"

interface Citation {
  page_title: string
  page_url: string
  source_type: string
}

interface MarkdownMessageProps {
  content: string
  citations?: Citation[]
  className?: string
}

function normalizeSourceLabel(value: string) {
  return value
    .replace(/\s+/g, " ")
    .replace(/[【】[\]]/g, "")
    .trim()
    .toLowerCase()
}

/**
 * Simple Markdown renderer for chat messages
 * Handles common markdown syntax without external dependencies
 */
export function MarkdownMessage({
  content,
  citations = [],
  className = ""
}: MarkdownMessageProps) {
  const citationByTitle = new Map(
    citations.map((citation) => [normalizeSourceLabel(citation.page_title), citation])
  )

  const renderMarkdown = (text: string): React.ReactNode[] => {
    const lines = text.split("\n")
    const elements: React.ReactNode[] = []
    let key = 0

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]

      // Empty line
      if (line.trim() === "") {
        elements.push(<br key={key++} />)
        continue
      }

      // Headers (## Header)
      const headerMatch = line.match(/^(#{1,6})\s+(.+)$/)
      if (headerMatch) {
        const level = headerMatch[1].length
        const text = headerMatch[2]
        const HeaderTag = `h${level}` as keyof JSX.IntrinsicElements
        const className =
          level === 1
            ? "plasmo-text-lg plasmo-font-bold plasmo-mb-2 plasmo-mt-3"
            : level === 2
              ? "plasmo-text-base plasmo-font-bold plasmo-mb-2 plasmo-mt-2"
              : "plasmo-text-sm plasmo-font-semibold plasmo-mb-1 plasmo-mt-2"

        elements.push(
          <HeaderTag key={key++} className={className}>
            {renderInlineMarkdown(text)}
          </HeaderTag>
        )
        continue
      }

      // Bullet points (• or - or *)
      const bulletMatch = line.match(/^(\s*)([-•*])\s+(.+)$/)
      if (bulletMatch) {
        const indent = bulletMatch[1].length
        const text = bulletMatch[3]
        elements.push(
          <div key={key++} className="plasmo-flex plasmo-gap-2 plasmo-mb-1" style={{ marginLeft: `${indent * 8}px` }}>
            <span className="plasmo-text-primary-600 plasmo-font-bold">•</span>
            <span className="plasmo-flex-1">{renderInlineMarkdown(text)}</span>
          </div>
        )
        continue
      }

      // Code blocks (```)
      if (line.trim().startsWith("```")) {
        const fenceMatch = line.trim().match(/^```([\w-]+)?/)
        const language = fenceMatch?.[1]?.toLowerCase() || ""
        const codeLines: string[] = []
        i++ // Skip opening ```
        while (i < lines.length && !lines[i].trim().startsWith("```")) {
          codeLines.push(lines[i])
          i++
        }

        const code = codeLines.join("\n")

        if (language === "mermaid") {
          elements.push(<MermaidDiagram key={key++} code={code} />)
          continue
        }

        elements.push(
          <pre key={key++} className="plasmo-bg-gray-100 plasmo-p-2 plasmo-rounded plasmo-text-xs plasmo-overflow-x-auto plasmo-my-2">
            <code>{code}</code>
          </pre>
        )
        continue
      }

      // Regular paragraph
      elements.push(
        <p key={key++} className="plasmo-mb-1">
          {renderInlineMarkdown(line)}
        </p>
      )
    }

    return elements
  }

  /**
   * Render inline markdown (bold, italic, code, links)
   */
  const renderInlineMarkdown = (text: string): React.ReactNode[] => {
    const parts: React.ReactNode[] = []
    let remaining = text
    let key = 0

    while (remaining.length > 0) {
      // Source annotation ([Source: ...])
      const sourceMatch = remaining.match(/^\[Source:\s*(.+?)\]/)
      if (sourceMatch) {
        const sourceLabel = sourceMatch[1]
        const normalizedLabel = normalizeSourceLabel(sourceLabel)
        const linkedCitation =
          citationByTitle.get(normalizedLabel) ||
          citations.find((citation) => {
            const normalizedCitation = normalizeSourceLabel(citation.page_title)
            return (
              normalizedCitation.includes(normalizedLabel) ||
              normalizedLabel.includes(normalizedCitation)
            )
          })

        const badgeClassName =
          "plasmo-inline-flex plasmo-items-center plasmo-gap-1 plasmo-rounded-full plasmo-border plasmo-border-slate-200 plasmo-bg-slate-50 plasmo-px-2 plasmo-py-0.5 plasmo-text-[11px] plasmo-font-medium plasmo-text-slate-600"

        const badgeContent = (
          <>
            <span className="plasmo-uppercase plasmo-tracking-[0.08em] plasmo-text-[10px] plasmo-font-semibold plasmo-text-slate-500">
              Source
            </span>
            <span className="plasmo-max-w-[220px] plasmo-truncate" title={sourceLabel}>
              {sourceLabel}
            </span>
          </>
        )

        parts.push(
          linkedCitation ? (
            <a
              key={key++}
              href={linkedCitation.page_url}
              target="_blank"
              rel="noopener noreferrer"
              className={`${badgeClassName} hover:plasmo-border-sky-300 hover:plasmo-bg-sky-50 hover:plasmo-text-sky-700`}
            >
              {badgeContent}
            </a>
          ) : (
            <span key={key++} className={badgeClassName}>
              {badgeContent}
            </span>
          )
        )
        remaining = remaining.slice(sourceMatch[0].length)
        continue
      }

      // Bold (**text**)
      const boldMatch = remaining.match(/^\*\*(.+?)\*\*/)
      if (boldMatch) {
        parts.push(
          <strong key={key++} className="plasmo-font-bold">
            {boldMatch[1]}
          </strong>
        )
        remaining = remaining.slice(boldMatch[0].length)
        continue
      }

      // Italic (*text*)
      const italicMatch = remaining.match(/^\*(.+?)\*/)
      if (italicMatch) {
        parts.push(
          <em key={key++} className="plasmo-italic">
            {italicMatch[1]}
          </em>
        )
        remaining = remaining.slice(italicMatch[0].length)
        continue
      }

      // Inline code (`code`)
      const codeMatch = remaining.match(/^`(.+?)`/)
      if (codeMatch) {
        parts.push(
          <code key={key++} className="plasmo-bg-gray-100 plasmo-px-1 plasmo-rounded plasmo-text-xs plasmo-font-mono">
            {codeMatch[1]}
          </code>
        )
        remaining = remaining.slice(codeMatch[0].length)
        continue
      }

      // Links ([text](url))
      const linkMatch = remaining.match(/^\[(.+?)\]\((.+?)\)/)
      if (linkMatch) {
        parts.push(
          <a
            key={key++}
            href={linkMatch[2]}
            target="_blank"
            rel="noopener noreferrer"
            className="plasmo-text-primary-600 plasmo-underline hover:plasmo-text-primary-700"
          >
            {linkMatch[1]}
          </a>
        )
        remaining = remaining.slice(linkMatch[0].length)
        continue
      }

      // Regular character
      parts.push(remaining[0])
      remaining = remaining.slice(1)
    }

    return parts
  }

  return (
    <div className={`plasmo-text-sm plasmo-leading-relaxed ${className}`}>
      {renderMarkdown(content)}
    </div>
  )
}
