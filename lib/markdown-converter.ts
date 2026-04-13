import TurndownService from "turndown"

/**
 * Configure Turndown service with Confluence-optimized rules
 */
export function createMarkdownConverter(baseUrl: string = window.location.href): TurndownService {
  const turndown = new TurndownService({
    headingStyle: "atx", // Use # for headings
    hr: "---",
    bulletListMarker: "-",
    codeBlockStyle: "fenced",
    fence: "```",
    emDelimiter: "*",
    strongDelimiter: "**",
    linkStyle: "inlined"
  })

  // Custom rule for code blocks
  turndown.addRule("confluenceCodeBlock", {
    filter: ["pre"],
    replacement: (content, node) => {
      const codeElement = (node as HTMLElement).querySelector("code")
      const language = codeElement?.getAttribute("data-language") || ""
      return `\n\`\`\`${language}\n${content.trim()}\n\`\`\`\n`
    }
  })

  // Custom rule for inline code
  turndown.addRule("inlineCode", {
    filter: (node) => {
      return (
        node.nodeName === "CODE" &&
        node.parentNode?.nodeName !== "PRE"
      )
    },
    replacement: (content) => {
      return `\`${content}\``
    }
  })

  // Custom rule for tables
  turndown.addRule("tables", {
    filter: "table",
    replacement: (content, node) => {
      const table = node as HTMLTableElement
      const rows = Array.from(table.querySelectorAll("tr"))

      if (rows.length === 0) return ""

      const markdown: string[] = []

      // Process header row
      const headerRow = rows[0]
      const headerCells = Array.from(headerRow.querySelectorAll("th, td"))
      const headers = headerCells.map((cell) => cell.textContent?.trim() || "")

      if (headers.length > 0) {
        markdown.push(`| ${headers.join(" | ")} |`)
        markdown.push(`| ${headers.map(() => "---").join(" | ")} |`)
      }

      // Process body rows
      const bodyRows = rows.slice(1)
      bodyRows.forEach((row) => {
        const cells = Array.from(row.querySelectorAll("td"))
        const cellContents = cells.map((cell) => cell.textContent?.trim() || "")
        if (cellContents.length > 0) {
          markdown.push(`| ${cellContents.join(" | ")} |`)
        }
      })

      return `\n${markdown.join("\n")}\n`
    }
  })

  // Custom rule for links - preserve full URLs
  turndown.addRule("links", {
    filter: "a",
    replacement: (content, node) => {
      const href = (node as HTMLAnchorElement).getAttribute("href")
      if (!href || href.startsWith("#")) return content

      // Convert relative URLs to absolute
      const absoluteUrl = new URL(href, baseUrl).toString()
      return `[${content}](${absoluteUrl})`
    }
  })

  // Remove empty paragraphs
  turndown.addRule("emptyParagraphs", {
    filter: (node) => {
      return (
        node.nodeName === "P" &&
        (!node.textContent || node.textContent.trim() === "")
      )
    },
    replacement: () => ""
  })

  return turndown
}

/**
 * Convert HTML to Markdown with Confluence-specific optimizations
 */
export function htmlToMarkdown(html: string, baseUrl?: string): string {
  const converter = createMarkdownConverter(baseUrl)
  let markdown = converter.turndown(html)

  // Post-processing: clean up extra whitespace
  markdown = markdown
    .replace(/\n{3,}/g, "\n\n") // Max 2 consecutive newlines
    .replace(/\n\s+\n/g, "\n\n") // Remove whitespace-only lines
    .trim()

  return markdown
}

/**
 * Estimate token count (rough approximation: 1 token ≈ 4 characters)
 */
export function estimateTokenCount(text: string): number {
  return Math.ceil(text.length / 4)
}
