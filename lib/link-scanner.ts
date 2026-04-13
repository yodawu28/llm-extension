/**
 * Link Scanner Service - Detect related documents in current page
 *
 * Automatically finds Confluence/Jira links in page content to suggest
 * proactive context gathering.
 */

export interface LinkedResource {
  url: string
  title: string
  type: 'confluence' | 'jira' | 'github' | 'doc' | 'generic'
  context: string  // Surrounding text for relevance
  confidence: 'high' | 'medium' | 'low'
}

export class LinkScanner {
  private static readonly CONFLUENCE_PATTERNS = [
    /atlassian\.net\/wiki/i,
    /confluence\./i,
    /wiki\./i,
    /\/display\//i,
    /\/pages\//i,
    /viewpage\.action/i
  ]

  private static readonly JIRA_PATTERNS = [
    /atlassian\.net\/browse/i,
    /jira\./i,
    /\/browse\/[A-Z]+-\d+/i
  ]

  private static readonly GITHUB_PATTERNS = [
    /github\.com\/.*\/(issues|pull|wiki|blob)/i
  ]

  private static readonly DOC_PATTERNS = [
    /docs\.google\.com/i,
    /notion\.so/i,
    /\/wiki\//i,
    /readme\.md$/i
  ]

  /**
   * Scan current page for related resource links
   */
  static scanCurrentPage(): LinkedResource[] {
    const links = this.extractAllLinks()
    const resources: LinkedResource[] = []

    for (const link of links) {
      const resource = this.analyzeLink(link)
      if (resource) {
        resources.push(resource)
      }
    }

    // Deduplicate by URL
    return this.deduplicateResources(resources)
  }

  /**
   * Extract all links from page
   */
  private static extractAllLinks(): HTMLAnchorElement[] {
    // Priority selectors for common platforms
    const prioritySelectors = [
      // Jira
      '[data-testid*="description"] a',
      '[data-testid*="comment"] a',
      '.issue-body a',

      // Confluence
      '.wiki-content a',
      '.page-content a',
      '[data-testid="content-area"] a',

      // GitHub
      '.markdown-body a',
      '.comment-body a',
      '.issue-title a',

      // Generic
      'main a',
      'article a',
      '[role="main"] a'
    ]

    const linkSet = new Set<HTMLAnchorElement>()

    for (const selector of prioritySelectors) {
      try {
        const elements = document.querySelectorAll<HTMLAnchorElement>(selector)
        elements.forEach(el => {
          if (el.href && !el.href.startsWith('javascript:')) {
            linkSet.add(el)
          }
        })
      } catch (e) {
        // Selector might not exist, continue
        console.debug(`Selector ${selector} not found`, e)
      }
    }

    return Array.from(linkSet)
  }

  /**
   * Analyze a link to determine if it's a relevant resource
   */
  private static analyzeLink(linkElement: HTMLAnchorElement): LinkedResource | null {
    const url = linkElement.href
    const text = linkElement.textContent?.trim() || ''
    const context = this.extractContext(linkElement)

    // Skip navigation/UI links
    if (this.isNavigationLink(linkElement)) {
      return null
    }

    // Skip already pinned (would need to check storage, for now skip same domain)
    if (url === window.location.href) {
      return null
    }

    // Determine type and confidence
    const type = this.detectLinkType(url)
    if (type === 'generic') {
      return null  // Only interested in documentation links
    }

    const confidence = this.calculateConfidence(linkElement, context)

    return {
      url,
      title: text || this.extractTitleFromURL(url),
      type,
      context,
      confidence
    }
  }

  /**
   * Detect link type based on URL patterns
   */
  private static detectLinkType(url: string): LinkedResource['type'] {
    if (this.CONFLUENCE_PATTERNS.some(pattern => pattern.test(url))) {
      return 'confluence'
    }
    if (this.JIRA_PATTERNS.some(pattern => pattern.test(url))) {
      return 'jira'
    }
    if (this.GITHUB_PATTERNS.some(pattern => pattern.test(url))) {
      return 'github'
    }
    if (this.DOC_PATTERNS.some(pattern => pattern.test(url))) {
      return 'doc'
    }
    return 'generic'
  }

  /**
   * Extract surrounding text context for relevance
   */
  private static extractContext(element: HTMLAnchorElement, maxLength: number = 150): string {
    const parent = element.closest('p, li, td, div')
    if (!parent) return ''

    const fullText = parent.textContent?.trim() || ''

    // Find position of link text within parent
    const linkText = element.textContent?.trim() || ''
    const linkPos = fullText.indexOf(linkText)

    if (linkPos === -1) return fullText.slice(0, maxLength)

    // Extract context around link (50 chars before, rest after)
    const start = Math.max(0, linkPos - 50)
    const end = Math.min(fullText.length, linkPos + linkText.length + maxLength - 50)

    let context = fullText.slice(start, end)

    // Add ellipsis if truncated
    if (start > 0) context = '...' + context
    if (end < fullText.length) context = context + '...'

    return context
  }

  /**
   * Check if link is navigation/UI element (not content)
   */
  private static isNavigationLink(element: HTMLAnchorElement): boolean {
    // Check if inside navigation elements
    const navParent = element.closest('nav, header, footer, aside, .navigation, .menu, .toolbar, .breadcrumbs')
    if (navParent) return true

    // Check link classes/attributes
    const navClasses = ['nav-link', 'menu-item', 'tab', 'button', 'btn']
    const classList = Array.from(element.classList)
    if (navClasses.some(cls => classList.some(c => c.includes(cls)))) {
      return true
    }

    // Check if it's a small icon/button link (likely UI)
    const text = element.textContent?.trim() || ''
    if (text.length === 0 || text.length === 1) {
      return true
    }

    return false
  }

  /**
   * Calculate confidence score based on link context
   */
  private static calculateConfidence(
    element: HTMLAnchorElement,
    context: string
  ): 'high' | 'medium' | 'low' {
    let score = 0

    // High confidence indicators
    const highConfidenceKeywords = [
      'documentation',
      'spec',
      'design',
      'architecture',
      'prd',
      'requirements',
      'implementation',
      'related',
      'see also',
      'refer to',
      'depends on'
    ]

    const contextLower = context.toLowerCase()
    const textLower = (element.textContent?.toLowerCase() || '')

    if (highConfidenceKeywords.some(kw => contextLower.includes(kw) || textLower.includes(kw))) {
      score += 2
    }

    // Medium confidence indicators
    const mediumConfidenceKeywords = [
      'ticket',
      'issue',
      'task',
      'bug',
      'feature',
      'page',
      'wiki'
    ]

    if (mediumConfidenceKeywords.some(kw => contextLower.includes(kw) || textLower.includes(kw))) {
      score += 1
    }

    // Link is in a list (often means related resources)
    if (element.closest('ul, ol')) {
      score += 1
    }

    // Link has descriptive text (not just "click here")
    const text = element.textContent?.trim() || ''
    if (text.length > 10 && !['click here', 'link', 'here'].includes(text.toLowerCase())) {
      score += 1
    }

    if (score >= 3) return 'high'
    if (score >= 1) return 'medium'
    return 'low'
  }

  /**
   * Extract readable title from URL if link text is not descriptive
   */
  private static extractTitleFromURL(url: string): string {
    try {
      const urlObj = new URL(url)

      // Jira ticket: Extract ticket key
      const jiraMatch = urlObj.pathname.match(/\/browse\/([A-Z]+-\d+)/)
      if (jiraMatch) {
        return jiraMatch[1]
      }

      // Confluence: Extract page title from path
      const confluenceMatch = urlObj.pathname.match(/\/([^\/]+)$/)
      if (confluenceMatch) {
        return decodeURIComponent(confluenceMatch[1]).replace(/[-_+]/g, ' ')
      }

      // GitHub: Extract repo/issue
      const githubMatch = urlObj.pathname.match(/\/([^\/]+\/[^\/]+)/)
      if (githubMatch) {
        return githubMatch[1]
      }

      // Fallback: Use hostname
      return urlObj.hostname
    } catch (e) {
      return url
    }
  }

  /**
   * Remove duplicate resources (same URL)
   */
  private static deduplicateResources(resources: LinkedResource[]): LinkedResource[] {
    const seen = new Map<string, LinkedResource>()

    for (const resource of resources) {
      const existing = seen.get(resource.url)

      // Keep higher confidence version
      if (!existing || this.compareConfidence(resource.confidence, existing.confidence) > 0) {
        seen.set(resource.url, resource)
      }
    }

    return Array.from(seen.values())
  }

  /**
   * Compare confidence levels (returns 1 if a > b, -1 if a < b, 0 if equal)
   */
  private static compareConfidence(a: string, b: string): number {
    const order = { high: 3, medium: 2, low: 1 }
    return order[a as keyof typeof order] - order[b as keyof typeof order]
  }

  /**
   * Filter resources to only high/medium confidence
   */
  static filterRelevant(resources: LinkedResource[]): LinkedResource[] {
    return resources.filter(r => r.confidence === 'high' || r.confidence === 'medium')
  }

  /**
   * Sort resources by confidence and type
   */
  static sortByRelevance(resources: LinkedResource[]): LinkedResource[] {
    return resources.sort((a, b) => {
      // First by confidence
      const confComp = this.compareConfidence(a.confidence, b.confidence)
      if (confComp !== 0) return -confComp  // Descending

      // Then by type priority (confluence > jira > github > doc)
      const typePriority = { confluence: 4, jira: 3, github: 2, doc: 1, generic: 0 }
      return typePriority[b.type] - typePriority[a.type]
    })
  }
}
