export {}

import notificationIconUrl from "url:~assets/icon128.png"

import { LinkScanner, type LinkedResource } from "~lib/link-scanner"
import { BackgroundFetcher, type FetchedResource } from "~lib/background-fetcher"
import { EXTENSION_BUILD_LABEL } from "~lib/build-info"
import { readUserSettings } from "~lib/user-settings"

// State management
let lastScannedURL: string | null = null
let scanInProgress = false
const SCAN_COOLDOWN_MS = 5000  // Don't rescan same page within 5 seconds
const DEFAULT_ACTION_TITLE = "Web Context Assistant"
const MAX_DIRECT_IMPORT_PAGES = 12
const activeImports = new Map<string, BackgroundFetcher>()
const STORAGE_PREFIXES = {
  fetched: "fetched_",
  discoveredLinks: "discovered_links_",
  page: "page_",
  diagramCache: "diagram_cache_"
} as const

function createNotificationId(prefix: string) {
  return `${prefix}-${Date.now()}-${crypto.randomUUID()}`
}

function clampImportResources(resources: LinkedResource[]) {
  return resources.slice(0, MAX_DIRECT_IMPORT_PAGES)
}

async function getPinnedPageUrlSet() {
  const result = await chrome.storage.local.get("pinnedPages")
  const pinnedPages = result.pinnedPages || []
  return new Set(
    pinnedPages
      .map((page: { url?: string }) => page.url)
      .filter((url: string | undefined): url is string => Boolean(url))
  )
}

async function filterPinnedResources(resources: LinkedResource[]) {
  const pinnedUrls = await getPinnedPageUrlSet()
  return resources.filter((resource) => !pinnedUrls.has(resource.url))
}

async function clearDiscoveredResources(tabId: number) {
  await chrome.storage.local.remove(`discovered_links_${tabId}`)
}

async function syncDiscoveredResourcesAfterImport(tabId: number, importedUrls: string[]) {
  if (importedUrls.length === 0) {
    return
  }

  const key = `discovered_links_${tabId}`
  const result = await chrome.storage.local.get(key)
  const discovered = result[key]

  if (!discovered?.resources?.length) {
    await clearMindReaderBadge(tabId)
    return
  }

  const importedUrlSet = new Set(importedUrls)
  const remainingResources = discovered.resources.filter(
    (resource: LinkedResource) => !importedUrlSet.has(resource.url)
  )

  if (remainingResources.length === discovered.resources.length) {
    return
  }

  if (remainingResources.length === 0) {
    await chrome.storage.local.remove(key)
    await clearMindReaderBadge(tabId)
    return
  }

  await chrome.storage.local.set({
    [key]: {
      ...discovered,
      resources: remainingResources,
      timestamp: Date.now()
    }
  })

  await setMindReaderBadge(tabId, remainingResources.length)
}

async function cancelActiveImports() {
  for (const fetcher of activeImports.values()) {
    fetcher.cancel()
  }

  activeImports.clear()
}

async function clearAllMindReaderBadges() {
  const tabs = await chrome.tabs.query({})

  await Promise.all(
    tabs
      .map((tab) => tab.id)
      .filter((tabId): tabId is number => tabId != null)
      .map((tabId) =>
        clearMindReaderBadge(tabId).catch(() => {
          return undefined
        })
      )
  )
}

async function clearStoredData(mode: "cache" | "all") {
  const allData = await chrome.storage.local.get(null)
  const keys = Object.keys(allData).filter((key) => {
    if (mode === "cache") {
      return (
        key.startsWith(STORAGE_PREFIXES.fetched) ||
        key.startsWith(STORAGE_PREFIXES.discoveredLinks) ||
        key.startsWith(STORAGE_PREFIXES.diagramCache)
      )
    }

    return (
      key === "pinnedPages" ||
      key.startsWith(STORAGE_PREFIXES.page) ||
      key.startsWith(STORAGE_PREFIXES.fetched) ||
      key.startsWith(STORAGE_PREFIXES.discoveredLinks) ||
      key.startsWith(STORAGE_PREFIXES.diagramCache)
    )
  })

  if (keys.length > 0) {
    await chrome.storage.local.remove(keys)
  }

  if (mode === "all") {
    lastScannedURL = null
    scanInProgress = false
    await cancelActiveImports()
  }

  await clearAllMindReaderBadges()

  return {
    removedCount: keys.length
  }
}

async function setMindReaderBadge(tabId: number, count: number) {
  await chrome.action.setBadgeBackgroundColor({
    color: "#2563eb",
    tabId
  })
  await chrome.action.setBadgeText({
    text: count > 99 ? "99+" : String(count),
    tabId
  })
  await chrome.action.setTitle({
    title: `Mind Reader: found ${count} related document${count > 1 ? "s" : ""}`,
    tabId
  })
}

async function clearMindReaderBadge(tabId: number) {
  await chrome.action.setBadgeText({
    text: "",
    tabId
  })
  await chrome.action.setTitle({
    title: DEFAULT_ACTION_TITLE,
    tabId
  })
}

async function showMindReaderStatusToast(
  tabId: number,
  tone: "loading" | "success" | "error",
  title: string,
  message: string,
  autoDismissMs: number = 4000,
  progress?: {
    completed: number
    total: number
    current?: string
  }
) {
  try {
    await chrome.tabs.sendMessage(tabId, {
      action: "show-mind-reader-status",
      payload: {
        tone,
        title,
        message,
        autoDismissMs,
        progress
      }
    })
  } catch (error) {
    console.warn("[Mind Reader] Could not show status toast:", error)
  }
}

async function showInPageMindReaderToast(tabId: number, resources: LinkedResource[]) {
  try {
    const response = await chrome.tabs.sendMessage(tabId, {
      action: "show-mind-reader-toast",
      payload: { resources }
    })

    return response?.success === true
  } catch (error) {
    console.warn("[Mind Reader] Could not show in-page toast:", error)
    return false
  }
}

async function initializeSidePanelBehavior() {
  try {
    await chrome.sidePanel.setPanelBehavior({
      openPanelOnActionClick: true
    })
  } catch (error) {
    console.warn("[Setup] Could not enable openPanelOnActionClick:", error)
  }
}

void initializeSidePanelBehavior()

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== "local" || !changes.userSettings) {
    return
  }

  const nextSettings = {
    ...(changes.userSettings.oldValue || {}),
    ...(changes.userSettings.newValue || {})
  }

  if (nextSettings.mindReaderScanMode === "manual" || nextSettings.mindReaderScanMode === "off") {
    lastScannedURL = null
    scanInProgress = false
    void clearAllMindReaderBadges()
  }
})

// When user clicks the extension icon, clear the badge.
// The actual panel opening is delegated to openPanelOnActionClick.
chrome.action.onClicked.addListener(async (tab) => {
  if (tab?.id != null) {
    await clearMindReaderBadge(tab.id)
  }
})

// Helper function to get active tab info
async function getActiveTabInfo(windowId: number) {
  const [tab] = await chrome.tabs.query({ active: true, windowId })
  if (!tab) return null
  return {
    tabId: tab.id ?? -1,
    title: tab.title || "Untitled",
    url: tab.url || "",
    hostname: tab.url ? new URL(tab.url).hostname : "",
    favIconUrl: tab.favIconUrl || ""
  }
}

// Broadcast active tab info to sidepanel
async function broadcastActiveTab(windowId: number) {
  const info = await getActiveTabInfo(windowId)
  if (!info) return

  // Send message to sidepanel (ignore errors if sidepanel is closed)
  try {
    await chrome.runtime.sendMessage({
      type: "ACTIVE_TAB_CHANGED",
      payload: info
    })
  } catch (e) {
    // Silently ignore - sidepanel may not be open
  }
}

// When user switches tab
chrome.tabs.onActivated.addListener(async (activeInfo: any) => {
  await broadcastActiveTab(activeInfo.windowId)
})

// When tab URL/title updates
chrome.tabs.onUpdated.addListener(async (tabId: any, changeInfo: any, tab: any) => {
  console.log(`[Tab Update] tabId=${tabId}, active=${tab.active}, status=${changeInfo.status}, url=${tab.url}`)

  if (!tab.active) {
    console.log(`[Tab Update] ⏭️ Skip - tab not active`)
    return
  }

  // Broadcast active tab info if URL/title changed
  if ((changeInfo.url || changeInfo.title || changeInfo.favIconUrl) && tab.windowId) {
    await broadcastActiveTab(tab.windowId)
  }

  if (changeInfo.url) {
    await clearMindReaderBadge(tabId)
  }

  // Mind Reader: Auto-scan for related links when page loads
  if (changeInfo.status === "complete" && tab.url) {
    console.log(`[Tab Update] ✅ Page complete, triggering Mind Reader`)
    await autoScanForRelatedLinks(tabId, tab.url)
  }
})

async function runMindReaderScan(
  tabId: number,
  url: string,
  options?: {
    force?: boolean
    mode?: "auto" | "manual"
  }
) {
  const scanMode = options?.mode || "auto"
  console.log(`[Mind Reader] Tab updated: ${url}`)

  if (!options?.force) {
    const userSettings = await readUserSettings()
    if (scanMode === "auto" && userSettings.mindReaderScanMode !== "auto") {
      console.log(`[Mind Reader] ⏭️ Skip - auto-scan disabled`)
      await clearMindReaderBadge(tabId)
      return {
        success: false,
        skipped: true,
        reason: "Auto-scan is disabled"
      }
    }

    if (scanMode === "manual" && userSettings.mindReaderScanMode === "off") {
      console.log(`[Mind Reader] ⏭️ Skip - manual scan disabled`)
      await clearMindReaderBadge(tabId)
      return {
        success: false,
        skipped: true,
        reason: "Mind Reader is turned off"
      }
    }
  }

  // Skip if already scanned recently
  if (!options?.force && lastScannedURL === url) {
    console.log(`[Mind Reader] ⏭️ Skip - already scanned: ${url}`)
    return {
      success: false,
      skipped: true,
      reason: "This page was already scanned recently"
    }
  }

  if (scanInProgress) {
    console.log(`[Mind Reader] ⏭️ Skip - scan in progress`)
    return {
      success: false,
      skipped: true,
      reason: "A scan is already in progress"
    }
  }

  // Only scan Jira/Confluence/GitHub pages
  if (!shouldScanPage(url)) {
    console.log(`[Mind Reader] ⏭️ Skip - page not in scan list`)
    await clearMindReaderBadge(tabId)
    await clearDiscoveredResources(tabId)
    return {
      success: false,
      skipped: true,
      reason: "This page type is not supported by Mind Reader"
    }
  }

  console.log(`[Mind Reader] 🔍 Starting scan for: ${url}`)
  scanInProgress = true
  lastScannedURL = url

  try {
    // Wait a bit for page to be ready
    await new Promise(resolve => setTimeout(resolve, 1500))

    // Execute link scanning directly in page context
    const [result] = await chrome.scripting.executeScript({
      target: { tabId },
      func: scanLinksInPage
    })

    if (result?.result && result.result.length > 0) {
      const resources: LinkedResource[] = result.result
      console.log(`[Mind Reader] ✅ Found ${resources.length} total resources`)

      // Filter to only high/medium confidence
      const relevant = LinkScanner.filterRelevant(resources)
      const freshResources = await filterPinnedResources(relevant)
      console.log(`[Mind Reader] 📊 After filtering: ${relevant.length} relevant (medium/high confidence)`)
      console.log(`[Mind Reader] 📌 After excluding pinned pages: ${freshResources.length} new resources`)

      if (freshResources.length > 0) {
        console.log(`[Mind Reader] 🔔 Showing notification for ${freshResources.length} resources`)

        // Show notification to user
        await showLinkSuggestionNotification(tabId, freshResources)

        // Store discovered links
        await chrome.storage.local.set({
          [`discovered_links_${tabId}`]: {
            url,
            resources: freshResources,
            timestamp: Date.now()
          }
        })
        return {
          success: true,
          foundCount: freshResources.length
        }
      } else {
        console.log(`[Mind Reader] ℹ️ No new related links found after excluding already imported pages`)
        await clearDiscoveredResources(tabId)
        await clearMindReaderBadge(tabId)
        return {
          success: true,
          foundCount: 0
        }
      }
    } else {
      console.log(`[Mind Reader] ℹ️ No links found on page`)
      await clearDiscoveredResources(tabId)
      await clearMindReaderBadge(tabId)
      return {
        success: true,
        foundCount: 0
      }
    }
  } catch (error) {
    console.error("[Mind Reader] ❌ Scan failed:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Scan failed"
    }
  } finally {
    scanInProgress = false

    // Reset cooldown after delay
    setTimeout(() => {
      if (lastScannedURL === url) {
        lastScannedURL = null
      }
    }, SCAN_COOLDOWN_MS)
  }
}

/**
 * Mind Reader: Automatically scan page for related documentation links
 */
async function autoScanForRelatedLinks(tabId: number, url: string) {
  await runMindReaderScan(tabId, url, { mode: "auto" })
}

async function manuallyScanCurrentTab(tabId: number, url: string) {
  return runMindReaderScan(tabId, url, { mode: "manual", force: true })
}

/**
 * Injected function to scan links in page context
 * This runs in the page's context, not extension context
 */
function scanLinksInPage(): LinkedResource[] {
  // Define types inline since this is injected
  interface LinkedResource {
    url: string
    title: string
    type: 'confluence' | 'jira' | 'github' | 'doc' | 'generic'
    context: string
    confidence: 'high' | 'medium' | 'low'
  }

  const CONFLUENCE_PATTERNS = [
    /atlassian\.net\/wiki/i,
    /confluence\./i,
    /\/display\//i,
    /\/pages\//i
  ]

  const JIRA_PATTERNS = [
    /atlassian\.net\/browse/i,
    /jira\./i,
    /\/browse\/[A-Z]+-\d+/i
  ]

  const GITHUB_PATTERNS = [
    /github\.com\/.*\/(issues|pull|wiki|blob)/i
  ]

  const DOC_PATTERNS = [
    /docs\.google\.com/i,
    /notion\.so/i,
    /\/wiki\//i
  ]

  function detectType(url: string): LinkedResource['type'] {
    if (CONFLUENCE_PATTERNS.some(p => p.test(url))) return 'confluence'
    if (JIRA_PATTERNS.some(p => p.test(url))) return 'jira'
    if (GITHUB_PATTERNS.some(p => p.test(url))) return 'github'
    if (DOC_PATTERNS.some(p => p.test(url))) return 'doc'
    return 'generic'
  }

  function isNavigationLink(element: HTMLAnchorElement): boolean {
    const navParent = element.closest('nav, header, footer, aside, .navigation, .menu, .toolbar, .breadcrumbs')
    if (navParent) return true

    const navClasses = ['nav-link', 'menu-item', 'tab', 'button', 'btn']
    const classList = Array.from(element.classList)
    if (navClasses.some(cls => classList.some(c => c.includes(cls)))) return true

    const text = element.textContent?.trim() || ''
    if (text.length === 0 || text.length === 1) return true

    return false
  }

  function extractContext(element: HTMLAnchorElement): string {
    const parent = element.closest('p, li, td, div')
    if (!parent) return ''

    const fullText = parent.textContent?.trim() || ''
    const linkText = element.textContent?.trim() || ''
    const linkPos = fullText.indexOf(linkText)

    if (linkPos === -1) return fullText.slice(0, 150)

    const start = Math.max(0, linkPos - 50)
    const end = Math.min(fullText.length, linkPos + linkText.length + 100)

    let context = fullText.slice(start, end)
    if (start > 0) context = '...' + context
    if (end < fullText.length) context = context + '...'

    return context
  }

  function calculateConfidence(element: HTMLAnchorElement, context: string): 'high' | 'medium' | 'low' {
    let score = 0

    const highKeywords = ['documentation', 'spec', 'design', 'architecture', 'prd', 'requirements', 'implementation', 'related', 'see also', 'refer to']
    const mediumKeywords = ['ticket', 'issue', 'task', 'bug', 'feature', 'page', 'wiki']

    const contextLower = context.toLowerCase()
    const textLower = (element.textContent?.toLowerCase() || '')

    if (highKeywords.some(kw => contextLower.includes(kw) || textLower.includes(kw))) score += 2
    if (mediumKeywords.some(kw => contextLower.includes(kw) || textLower.includes(kw))) score += 1
    if (element.closest('ul, ol')) score += 1

    const text = element.textContent?.trim() || ''
    if (text.length > 10 && !['click here', 'link', 'here'].includes(text.toLowerCase())) score += 1

    if (score >= 3) return 'high'
    if (score >= 1) return 'medium'
    return 'low'
  }

  // Extract all links
  const selectors = [
    '[data-testid*="description"] a',
    '[data-testid*="comment"] a',
    '.wiki-content a',
    '.page-content a',
    '.markdown-body a',
    'main a',
    'article a'
  ]

  const linkSet = new Set<HTMLAnchorElement>()

  for (const selector of selectors) {
    try {
      const elements = document.querySelectorAll<HTMLAnchorElement>(selector)
      elements.forEach(el => {
        if (el.href && !el.href.startsWith('javascript:')) {
          linkSet.add(el)
        }
      })
    } catch (e) {
      // Selector might not exist
    }
  }

  const resources: LinkedResource[] = []
  const seen = new Set<string>()

  for (const link of linkSet) {
    if (isNavigationLink(link)) continue
    if (link.href === window.location.href) continue

    const type = detectType(link.href)
    if (type === 'generic') continue

    if (seen.has(link.href)) continue
    seen.add(link.href)

    const context = extractContext(link)
    const confidence = calculateConfidence(link, context)

    // Only return medium/high confidence
    if (confidence === 'low') continue

    resources.push({
      url: link.href,
      title: link.textContent?.trim() || link.href,
      type,
      context,
      confidence
    })
  }

  return resources
}

/**
 * Check if page should be scanned for links
 */
function shouldScanPage(url: string): boolean {
  const scanPatterns = [
    /atlassian\.net/i,
    /confluence/i,
    /jira/i,
    /wiki\./i,
    /viewpage\.action/i,
    /\/browse\/[A-Z]+-\d+/i,
    /github\.com/i,
    /notion\.so/i,
    /test-mindreader\.html$/i  // For testing
  ]

  return scanPatterns.some(pattern => pattern.test(url))
}

/**
 * Show notification with discovered links
 */
async function showLinkSuggestionNotification(tabId: number, resources: LinkedResource[]) {
  const count = resources.length
  const types = [...new Set(resources.map(r => r.type))]
  const userSettings = await readUserSettings()

  await setMindReaderBadge(tabId, count)

  let shownInPage = false

  if (userSettings.mindReaderPopupsEnabled) {
    shownInPage = await showInPageMindReaderToast(tabId, resources)
  }

  if (userSettings.mindReaderPopupsEnabled && !shownInPage) {
    await chrome.notifications.create(
      createNotificationId("link-suggestions"),
      {
        type: "basic",
        iconUrl: notificationIconUrl,
        title: `Found ${count} related document${count > 1 ? "s" : ""}`,
        message: `Detected ${types.join(", ")} links. Click to add to context.`,
        buttons: [
          { title: "Add All" },
          { title: "Use Extension Icon" }
        ],
        priority: 1
      }
    )
  }

  // Also broadcast to sidepanel
  try {
    await chrome.runtime.sendMessage({
      type: "LINKS_DISCOVERED",
      payload: {
        resources,
        count
      }
    })
  } catch (e) {
    // Sidepanel may not be open
  }
}

// Handle notification button clicks
chrome.notifications.onButtonClicked.addListener(async (notificationId, buttonIndex) => {
  // Get active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
  if (!tab?.id) return

  // Get discovered links
  const data = await chrome.storage.local.get(`discovered_links_${tab.id}`)
  const discovered = data[`discovered_links_${tab.id}`]

  if (!discovered) return

  const resources: LinkedResource[] = discovered.resources

  if (buttonIndex === 0) {
    // Add All
    await addAllResourcestoContext(resources, tab.id)
  } else if (buttonIndex === 1) {
    await showMindReaderStatusToast(
      tab.id,
      "error",
      "Open the panel from the extension icon",
      "Chrome only allows the side panel to open from a direct extension click. Click the extension icon to open it.",
      5000
    )
  }

  // Clear notification
  await clearMindReaderBadge(tab.id)
  await chrome.notifications.clear(notificationId)
})

/**
 * Add all discovered resources to context
 */
async function addAllResourcestoContext(
  resources: LinkedResource[],
  sourceTabId?: number,
  importId?: string
) {
  const unpinnedResources = await filterPinnedResources(resources)
  const limitedResources = clampImportResources(unpinnedResources)
  const fetcher = new BackgroundFetcher()

  if (importId) {
    activeImports.set(importId, fetcher)
  }

  try {
    if (limitedResources.length === 0) {
      throw new Error("All selected pages are already in context")
    }

    // Fetch with progress updates
    const results = await fetcher.fetchResources(limitedResources, (progress) => {
      if (sourceTabId != null) {
        void showMindReaderStatusToast(
          sourceTabId,
          "loading",
          "Importing related pages",
          progress.current
            ? `Fetching ${progress.current}`
            : `Imported ${progress.completed} of ${progress.total} pages`,
          0,
          {
            completed: progress.completed,
            total: progress.total,
            current: progress.current
          }
        )
      }
    })

    // Filter successful fetches
    const successful = results.filter(r => r.status === "success")
    const cancelled = fetcher.wasCancelled()
    let addedCount = 0

    if (successful.length > 0) {
      // Add to pinned pages
      const addResult = await addFetchedResourcesToPinnedPages(successful)
      addedCount = addResult.addedCount

      if (sourceTabId != null && addResult.addedUrls.length > 0) {
        await syncDiscoveredResourcesAfterImport(sourceTabId, addResult.addedUrls)
      }

      if (sourceTabId != null) {
        await showMindReaderStatusToast(
          sourceTabId,
          cancelled ? "error" : "success",
          cancelled ? "Import cancelled" : "Pages added to context",
          cancelled
            ? `Added ${addResult.addedCount} page${addResult.addedCount > 1 ? "s" : ""} before cancellation.`
            : `Added ${addResult.addedCount} of ${limitedResources.length} related page${limitedResources.length > 1 ? "s" : ""}.`,
          3500
        )
      }
    }

    if (successful.length === 0) {
      throw new Error(cancelled ? "Import cancelled" : "No related documents could be fetched")
    }

    if (addedCount === 0) {
      throw new Error("All selected pages are already in context")
    }

    return {
      attemptedCount: limitedResources.length,
      addedCount,
      cancelled,
      truncated: unpinnedResources.length > limitedResources.length
    }
  } catch (error) {
    console.error("[Mind Reader] Add all failed:", error)

    if (sourceTabId != null) {
      await showMindReaderStatusToast(
        sourceTabId,
        "error",
        "Could not add pages",
        error instanceof Error ? error.message : "Fetching related documents failed. Try again.",
        5000
      )
    }

    throw error
  } finally {
    if (importId) {
      activeImports.delete(importId)
    }
  }
}

/**
 * Add fetched resources to pinned pages
 */
async function addFetchedResourcesToPinnedPages(resources: FetchedResource[]) {
  // Get existing pinned pages
  const result = await chrome.storage.local.get("pinnedPages")
  const pinnedPages = result.pinnedPages || []
  const pinnedUrls = new Set(
    pinnedPages
      .map((page: { url?: string }) => page.url)
      .filter((url: string | undefined): url is string => Boolean(url))
  )
  const newResources = resources.filter((resource) => !pinnedUrls.has(resource.url))

  const newPages = newResources.map(resource => ({
    id: crypto.randomUUID(),
    title: resource.title,
    url: resource.url,
    tokenEstimate: resource.tokenEstimate || 0,
    addedAt: Date.now(),
    favicon: undefined  // Will be fetched by frontend
  }))

  // Store markdown separately
  for (const resource of newResources) {
    const pageId = newPages.find(p => p.url === resource.url)?.id
    if (pageId && resource.markdown) {
      await chrome.storage.local.set({
        [`page_${pageId}`]: {
          markdown: resource.markdown,
          metadata: {
            tokenEstimate: resource.tokenEstimate,
            fetchedAt: resource.fetchedAt
          }
        }
      })
    }
  }

  // Update pinned pages
  await chrome.storage.local.set({
    pinnedPages: [...pinnedPages, ...newPages]
  })

  return {
    addedCount: newPages.length,
    addedUrls: newPages.map((page) => page.url)
  }
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "offscreen-parse-document") {
    return false
  }

  if (message.type === "LINKS_SCANNED") {
    // Content script has scanned and sent results
    const resources: LinkedResource[] = message.payload.resources

    if (resources.length > 0 && sender.tab?.id != null) {
      showLinkSuggestionNotification(sender.tab.id, resources)
    }

    sendResponse({ success: true })
    return false
  }

  if (message.action === "mind-reader-add-all") {
    ;(async () => {
      try {
        const resources: LinkedResource[] = message.payload?.resources || []
        const importId: string | undefined = message.payload?.importId
        const sourceTabId: number | undefined = message.payload?.sourceTabId ?? sender.tab?.id
        const result = await addAllResourcestoContext(resources, sourceTabId, importId)

        try {
          await chrome.runtime.sendMessage({
            type: "PAGES_ADDED",
            payload: {
              count: result.addedCount,
              tabId: sourceTabId
            }
          })
        } catch (e) {
          // Sidepanel may not be open
        }

        if (sourceTabId != null) {
          const key = `discovered_links_${sourceTabId}`
          const discovered = await chrome.storage.local.get(key)

          if (!discovered[key]?.resources?.length) {
            await clearMindReaderBadge(sourceTabId)
          }
        }

        sendResponse({
          success: true,
          addedCount: result.addedCount,
          attemptedCount: result.attemptedCount,
          cancelled: result.cancelled,
          truncated: result.truncated,
          maxAllowed: MAX_DIRECT_IMPORT_PAGES
        })
      } catch (error) {
        sendResponse({
          success: false,
          error: error instanceof Error ? error.message : "Failed to add resources"
        })
      }
    })()
    return true
  }

  if (message.action === "mind-reader-cancel-import") {
    ;(async () => {
      try {
        const importId: string | undefined = message.payload?.importId
        const fetcher = importId ? activeImports.get(importId) : undefined

        if (fetcher) {
          fetcher.cancel()
        }

        sendResponse({
          success: true,
          cancelled: Boolean(fetcher)
        })
      } catch (error) {
        sendResponse({
          success: false,
          error: error instanceof Error ? error.message : "Failed to cancel import"
        })
      }
    })()
    return true
  }

  if (message.action === "mind-reader-open-panel") {
    ;(async () => {
      try {
        if (sender.tab?.id != null) {
          await clearMindReaderBadge(sender.tab.id)
        }

        if (sender.tab?.id != null) {
          await showMindReaderStatusToast(
            sender.tab.id,
            "error",
            "Open the panel from the extension icon",
            "Chrome blocks side-panel opening here. Click the extension icon to open it.",
            5000
          )
        }

        sendResponse({
          success: false,
          requiresUserGesture: true,
          error: "Click the extension icon to open the side panel"
        })
      } catch (error) {
        sendResponse({
          success: false,
          error: error instanceof Error ? error.message : "Failed to open side panel"
        })
      }
    })()
    return true
  }

  if (message.action === "mind-reader-dismiss") {
    ;(async () => {
      try {
        if (sender.tab?.id != null) {
          await clearMindReaderBadge(sender.tab.id)
        }

        sendResponse({ success: true })
      } catch (error) {
        sendResponse({
          success: false,
          error: error instanceof Error ? error.message : "Failed to dismiss toast"
        })
      }
    })()
    return true
  }

  if (message.action === "mind-reader-scan-current-tab") {
    ;(async () => {
      try {
        const targetTabId: number | undefined = message.payload?.tabId ?? sender.tab?.id

        if (targetTabId == null) {
          throw new Error("No target tab available for manual Mind Reader scan")
        }

        const tab = await chrome.tabs.get(targetTabId)
        if (!tab.url) {
          throw new Error("Current tab URL is not available")
        }

        const result = await manuallyScanCurrentTab(targetTabId, tab.url)
        sendResponse({
          success: result.success,
          foundCount: result.foundCount || 0,
          skipped: result.skipped || false,
          reason: result.reason,
          error: result.error
        })
      } catch (error) {
        sendResponse({
          success: false,
          error: error instanceof Error ? error.message : "Manual scan failed"
        })
      }
    })()
    return true
  }

  if (message.action === "clear-import-cache") {
    ;(async () => {
      try {
        const result = await clearStoredData("cache")
        sendResponse({
          success: true,
          removedCount: result.removedCount
        })
      } catch (error) {
        sendResponse({
          success: false,
          error: error instanceof Error ? error.message : "Failed to clear cache"
        })
      }
    })()
    return true
  }

  if (message.action === "reset-local-data") {
    ;(async () => {
      try {
        const result = await clearStoredData("all")
        sendResponse({
          success: true,
          removedCount: result.removedCount
        })
      } catch (error) {
        sendResponse({
          success: false,
          error: error instanceof Error ? error.message : "Failed to reset local data"
        })
      }
    })()
    return true
  }

  return false
})

console.log(`[Web Context Assistant] Background service worker loaded with Mind Reader (${EXTENSION_BUILD_LABEL})`)
console.log("[Setup] ✅ Tab update listener registered")
console.log("[Setup] ✅ Tab activated listener registered")
console.log("[Setup] ✅ Action click listener registered")
