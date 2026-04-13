/**
 * Background Fetcher Service - Fetch and scrape linked resources
 *
 * Handles background fetching of discovered links without blocking UI.
 * Includes rate limiting, error handling, and caching.
 */

import type { LinkedResource } from "./link-scanner"
import type { ScrapedContent } from "./page-content-extractor"

export interface FetchedResource extends LinkedResource {
  markdown?: string
  tokenEstimate?: number
  fetchedAt: number
  status: 'pending' | 'fetching' | 'success' | 'error'
  error?: string
}

export interface FetchProgress {
  total: number
  completed: number
  current?: string
  status: 'idle' | 'fetching' | 'complete' | 'error'
}

export class BackgroundFetcher {
  private static readonly CACHE_TTL_MS = 24 * 60 * 60 * 1000  // 24 hours
  private static readonly RATE_LIMIT_MS = 1000  // 1 second between requests
  private static readonly MAX_CONCURRENT = 1
  private static readonly OFFSCREEN_DOCUMENT_PATH = "tabs/offscreen.html"
  private static readonly MIN_FETCH_MARKDOWN_LENGTH = 160
  private static offscreenReadyPromise?: Promise<void>

  private fetchQueue: LinkedResource[] = []
  private activeRequests = 0
  private progressCallback?: (progress: FetchProgress) => void
  private workerTabId?: number
  private workerWindowId?: number
  private cancelled = false

  /**
   * Fetch multiple resources with rate limiting
   */
  async fetchResources(
    resources: LinkedResource[],
    onProgress?: (progress: FetchProgress) => void
  ): Promise<FetchedResource[]> {
    this.cancelled = false
    this.progressCallback = onProgress
    this.fetchQueue = [...resources]

    const results: FetchedResource[] = []
    // Check cache first
    const cachedResults = await this.checkCache(resources)
    const uncached = resources.filter(r => !cachedResults.has(r.url))

    // Add cached to results
    for (const [url, cached] of cachedResults) {
      results.push(cached)
    }

    // Update progress
    this.notifyProgress({
      total: resources.length,
      completed: cachedResults.size,
      status: uncached.length > 0 ? 'fetching' : 'complete'
    })

    try {
      for (const resource of uncached) {
        if (this.cancelled) {
          break
        }

        const result = await this.fetchWithRetry(resource)
        results.push(result)
        this.notifyProgress({
          total: resources.length,
          completed: results.length,
          current: resource.title,
          status: 'fetching'
        })
      }
    } finally {
      await this.cleanupWorkerTab()
    }

    this.notifyProgress({
      total: resources.length,
      completed: results.length,
      status: 'complete'
    })

    // Cache successful results
    await this.cacheResults(results.filter(r => r.status === 'success'))

    return results
  }

  /**
   * Fetch a single resource with retry logic
   */
  private async fetchWithRetry(
    resource: LinkedResource,
    retries: number = 2
  ): Promise<FetchedResource> {
    const fetchedResource: FetchedResource = {
      ...resource,
      fetchedAt: Date.now(),
      status: 'fetching'
    }

    for (let attempt = 0; attempt <= retries; attempt++) {
      if (this.cancelled) {
        fetchedResource.status = 'error'
        fetchedResource.error = 'Import cancelled'
        return fetchedResource
      }

      try {
        // Rate limiting
        if (this.activeRequests >= BackgroundFetcher.MAX_CONCURRENT) {
          await this.wait(BackgroundFetcher.RATE_LIMIT_MS)
        }

        this.activeRequests++
        let result: Awaited<ReturnType<typeof this.scrapeURL>>

        try {
          result = await this.scrapeURL(resource.url)
        } finally {
          this.activeRequests = Math.max(0, this.activeRequests - 1)
        }

        if (result.success) {
          fetchedResource.markdown = result.markdown
          fetchedResource.tokenEstimate = result.tokenEstimate
          fetchedResource.status = 'success'
          return fetchedResource
        } else {
          throw new Error(result.error || 'Failed to scrape')
        }
      } catch (error) {
        if (attempt === retries) {
          // Final attempt failed
          fetchedResource.status = 'error'
          fetchedResource.error = error instanceof Error ? error.message : 'Unknown error'
          return fetchedResource
        }

        // Wait before retry
        await this.wait(BackgroundFetcher.RATE_LIMIT_MS * (attempt + 1))
      }
    }

    return fetchedResource
  }

  /**
   * Scrape URL by opening in hidden tab (respects same-origin policy)
   */
  private async scrapeURL(url: string): Promise<{
    success: boolean
    markdown?: string
    tokenEstimate?: number
    error?: string
  }> {
    const fetchResult = await this.scrapeURLWithFetch(url)

    if (fetchResult.success) {
      return fetchResult
    }

    console.info(`[Mind Reader] Falling back to worker tab for ${url}: ${fetchResult.error}`)

    return this.scrapeURLWithWorkerTab(url)
  }

  private async scrapeURLWithFetch(url: string): Promise<{
    success: boolean
    markdown?: string
    tokenEstimate?: number
    error?: string
  }> {
    if (this.cancelled) {
      return {
        success: false,
        error: "Import cancelled"
      }
    }

    try {
      const response = await fetch(url, {
        credentials: "include",
        redirect: "follow"
      })

      if (!response.ok) {
        return {
          success: false,
          error: `Fetch failed with status ${response.status}`
        }
      }

      const contentType = response.headers.get("content-type") || ""
      const isHtmlResponse =
        contentType.includes("text/html") || contentType.includes("application/xhtml+xml")

      if (!isHtmlResponse) {
        return {
          success: false,
          error: `Unsupported content type: ${contentType || "unknown"}`
        }
      }

      const html = await response.text()

      if (!html.trim()) {
        return {
          success: false,
          error: "Fetched document was empty"
        }
      }

      const finalUrl = response.url || url
      const parsed = await this.parseFetchedHtml(finalUrl, html)

      if (!parsed.success || !parsed.data) {
        return {
          success: false,
          error: parsed.error || "Offscreen parsing failed"
        }
      }

      if (this.shouldFallbackToWorker(parsed.data)) {
        return {
          success: false,
          error: "Fetched HTML requires live-tab fallback"
        }
      }

      console.info(`[Mind Reader] Imported via zero-visible fetch: ${finalUrl}`)

      return {
        success: true,
        markdown: parsed.data.markdownContent,
        tokenEstimate: parsed.data.metadata.tokenEstimate
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    }
  }

  private async scrapeURLWithWorkerTab(url: string): Promise<{
    success: boolean
    markdown?: string
    tokenEstimate?: number
    error?: string
  }> {
    try {
      const tabId = await this.getOrCreateWorkerTab()
      await chrome.tabs.update(tabId, { url, active: false })

      await this.waitForTabLoad(tabId)

      const response = await chrome.tabs.sendMessage(tabId, {
        action: "scrape-page"
      })

      if (response?.success) {
        return {
          success: true,
          markdown: response.data.markdownContent,
          tokenEstimate: response.data.metadata.tokenEstimate
        }
      }

      return {
        success: false,
        error: "Scraping failed"
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error"
      }
    }
  }

  private shouldFallbackToWorker(scraped: ScrapedContent): boolean {
    const normalized = `${scraped.title}\n${scraped.markdownContent}`.toLowerCase()
    const authShellKeywords = [
      "log in",
      "login",
      "sign in",
      "access denied",
      "permission denied",
      "unauthorized",
      "forbidden",
      "session expired"
    ]

    if (scraped.markdownContent.trim().length < BackgroundFetcher.MIN_FETCH_MARKDOWN_LENGTH) {
      return true
    }

    if (scraped.metadata.wordCount < 40) {
      return true
    }

    return authShellKeywords.some((keyword) => normalized.includes(keyword))
  }

  private async parseFetchedHtml(
    url: string,
    html: string
  ): Promise<{
    success: boolean
    data?: ScrapedContent
    error?: string
  }> {
    await BackgroundFetcher.ensureOffscreenDocument()

    const response = await chrome.runtime.sendMessage({
      action: "offscreen-parse-document",
      payload: {
        url,
        html
      }
    })

    if (response?.success && response.data) {
      return {
        success: true,
        data: response.data as ScrapedContent
      }
    }

    return {
      success: false,
      error: response?.error || "Offscreen parser returned no data"
    }
  }

  private static async ensureOffscreenDocument(): Promise<void> {
    const offscreenUrl = chrome.runtime.getURL(BackgroundFetcher.OFFSCREEN_DOCUMENT_PATH)
    const contexts = await chrome.runtime.getContexts({
      contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
      documentUrls: [offscreenUrl]
    })

    if (contexts.length > 0) {
      return
    }

    if (!BackgroundFetcher.offscreenReadyPromise) {
      BackgroundFetcher.offscreenReadyPromise = chrome.offscreen
        .createDocument({
          url: BackgroundFetcher.OFFSCREEN_DOCUMENT_PATH,
          reasons: [chrome.offscreen.Reason.DOM_PARSER],
          justification: "Parse fetched HTML into markdown without opening visible tabs"
        })
        .finally(() => {
          BackgroundFetcher.offscreenReadyPromise = undefined
        })
    }

    await BackgroundFetcher.offscreenReadyPromise
  }

  /**
   * Wait for tab to finish loading
   */
  private waitForTabLoad(tabId: number, timeout: number = 30000): Promise<void> {
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timeoutId)
        chrome.tabs.onUpdated.removeListener(listener)
        chrome.tabs.onRemoved.removeListener(removeListener)
      }

      const timeoutId = setTimeout(() => {
        cleanup()
        reject(new Error('Tab load timeout'))
      }, timeout)

      const listener = (updatedTabId: number, changeInfo: chrome.tabs.TabChangeInfo) => {
        if (updatedTabId === tabId && changeInfo.status === 'complete') {
          cleanup()
          // Give content script time to initialize
          setTimeout(resolve, 500)
        }
      }

      const removeListener = (removedTabId: number) => {
        if (removedTabId === tabId) {
          cleanup()
          reject(new Error('Tab removed before load completed'))
        }
      }

      chrome.tabs.onUpdated.addListener(listener)
      chrome.tabs.onRemoved.addListener(removeListener)
    })
  }

  private async getOrCreateWorkerTab(): Promise<number> {
    if (this.workerTabId != null) {
      try {
        const existingTab = await chrome.tabs.get(this.workerTabId)
        if (existingTab.id != null) {
          return existingTab.id
        }
      } catch (error) {
        this.workerTabId = undefined
      }
    }

    if (this.workerWindowId == null) {
      const workerWindow = await chrome.windows.create({
        url: "about:blank",
        focused: false,
        type: "popup",
        state: "minimized"
      })
      this.workerWindowId = workerWindow.id

      const existingTab = workerWindow.tabs?.[0]
      if (existingTab?.id != null) {
        this.workerTabId = existingTab.id
        return existingTab.id
      }
    }

    const tab = await chrome.tabs.create({
      url: "about:blank",
      active: false,
      windowId: this.workerWindowId
    })

    if (!tab.id) {
      throw new Error("Failed to create background worker tab")
    }

    this.workerTabId = tab.id
    return tab.id
  }

  private async cleanupWorkerTab(): Promise<void> {
    if (this.workerTabId != null) {
      try {
        await chrome.tabs.remove(this.workerTabId)
      } catch (error) {
        // Ignore cleanup failures
      }
      this.workerTabId = undefined
    }

    if (this.workerWindowId != null) {
      try {
        await chrome.windows.remove(this.workerWindowId)
      } catch (error) {
        // Ignore cleanup failures
      }
      this.workerWindowId = undefined
    }
  }

  /**
   * Check cache for previously fetched resources
   */
  private async checkCache(resources: LinkedResource[]): Promise<Map<string, FetchedResource>> {
    const cached = new Map<string, FetchedResource>()

    try {
      const keys = resources.map(r => `fetched_${this.hashURL(r.url)}`)
      const results = await chrome.storage.local.get(keys)

      for (const resource of resources) {
        const key = `fetched_${this.hashURL(resource.url)}`
        const cachedData = results[key]

        if (cachedData) {
          // Check if cache is still valid
          const age = Date.now() - cachedData.fetchedAt
          if (age < BackgroundFetcher.CACHE_TTL_MS) {
            cached.set(resource.url, cachedData)
          }
        }
      }
    } catch (error) {
      console.error('Cache check failed:', error)
    }

    return cached
  }

  /**
   * Cache fetched results
   */
  private async cacheResults(results: FetchedResource[]): Promise<void> {
    try {
      const cacheData: Record<string, FetchedResource> = {}

      for (const result of results) {
        const key = `fetched_${this.hashURL(result.url)}`
        cacheData[key] = result
      }

      await chrome.storage.local.set(cacheData)
    } catch (error) {
      console.error('Cache save failed:', error)
    }
  }

  /**
   * Simple URL hash for cache keys
   */
  private hashURL(url: string): string {
    let hash = 0
    for (let i = 0; i < url.length; i++) {
      const char = url.charCodeAt(i)
      hash = ((hash << 5) - hash) + char
      hash = hash & hash  // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36)
  }

  /**
   * Notify progress callback
   */
  private notifyProgress(progress: FetchProgress): void {
    if (this.progressCallback) {
      this.progressCallback(progress)
    }
  }

  /**
   * Wait utility
   */
  private wait(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  /**
   * Cancel ongoing fetches
   */
  cancel(): void {
    this.cancelled = true
    this.fetchQueue = []
    void this.cleanupWorkerTab()
  }

  wasCancelled(): boolean {
    return this.cancelled
  }

  /**
   * Clear cache
   */
  static async clearCache(): Promise<void> {
    const allData = await chrome.storage.local.get(null)
    const cacheKeys = Object.keys(allData).filter(k => k.startsWith('fetched_'))

    if (cacheKeys.length > 0) {
      await chrome.storage.local.remove(cacheKeys)
    }
  }
}
