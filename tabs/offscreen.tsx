import { extractScrapedContentFromHTML } from "~lib/page-content-extractor"

declare global {
  interface Window {
    __mindReaderOffscreenReady?: boolean
  }
}

if (!window.__mindReaderOffscreenReady) {
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.action !== "offscreen-parse-document") {
      return false
    }

    try {
      const html = message.payload?.html
      const url = message.payload?.url
      const title = message.payload?.title

      if (typeof html !== "string" || typeof url !== "string") {
        sendResponse({
          success: false,
          error: "Missing html or url for offscreen parsing"
        })
        return false
      }

      const scraped = extractScrapedContentFromHTML(html, url, title)

      if (!scraped) {
        sendResponse({
          success: false,
          error: "No main content found in fetched document"
        })
        return false
      }

      sendResponse({
        success: true,
        data: scraped
      })
    } catch (error) {
      sendResponse({
        success: false,
        error: error instanceof Error ? error.message : "Offscreen parsing failed"
      })
    }

    return false
  })

  window.__mindReaderOffscreenReady = true
  console.log("[Mind Reader] Offscreen parser ready")
}

function OffscreenPage() {
  return null
}

export default OffscreenPage
