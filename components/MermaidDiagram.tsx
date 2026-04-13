import { useEffect, useId, useRef, useState } from "react"
import mermaid from "mermaid"

interface MermaidDiagramProps {
  code: string
}

let mermaidInitialized = false

function ensureMermaidInitialized() {
  if (mermaidInitialized) return

  mermaid.initialize({
    startOnLoad: false,
    theme: "neutral",
    securityLevel: "loose",
    fontFamily: "inherit"
  })

  mermaidInitialized = true
}

export function MermaidDiagram({ code }: MermaidDiagramProps) {
  const diagramId = useId().replace(/:/g, "-")
  const [svg, setSvg] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [showCode, setShowCode] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  function showStatus(message: string) {
    setStatusMessage(message)

    if (statusTimerRef.current) {
      clearTimeout(statusTimerRef.current)
    }

    statusTimerRef.current = setTimeout(() => {
      setStatusMessage(null)
      statusTimerRef.current = null
    }, 2200)
  }

  function getSvgDimensions(svgMarkup: string) {
    const parsed = new DOMParser().parseFromString(svgMarkup, "image/svg+xml")
    const svgElement = parsed.querySelector("svg")

    if (!svgElement) {
      return { width: 1200, height: 800 }
    }

    const widthAttr = Number(svgElement.getAttribute("width"))
    const heightAttr = Number(svgElement.getAttribute("height"))

    if (Number.isFinite(widthAttr) && Number.isFinite(heightAttr) && widthAttr > 0 && heightAttr > 0) {
      return { width: widthAttr, height: heightAttr }
    }

    const viewBox = svgElement.getAttribute("viewBox")?.split(/\s+/).map(Number)
    if (viewBox && viewBox.length === 4) {
      const [, , width, height] = viewBox
      if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
        return { width, height }
      }
    }

    return { width: 1200, height: 800 }
  }

  function downloadBlob(blob: Blob, filename: string) {
    const blobUrl = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = blobUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()

    setTimeout(() => {
      URL.revokeObjectURL(blobUrl)
    }, 1000)
  }

  async function handleCopyCode() {
    try {
      await navigator.clipboard.writeText(code)
      showStatus("Copied Mermaid")
    } catch (error) {
      console.error("[MermaidDiagram] Failed to copy Mermaid code:", error)
      showStatus("Copy failed")
    }
  }

  function handleExportSvg() {
    if (!svg) {
      showStatus("No SVG available")
      return
    }

    downloadBlob(
      new Blob([svg], { type: "image/svg+xml;charset=utf-8" }),
      `visual-architect-${diagramId}.svg`
    )
    showStatus("Exported SVG")
  }

  async function handleExportPng() {
    if (!svg) {
      showStatus("No PNG available")
      return
    }

    try {
      const { width, height } = getSvgDimensions(svg)
      const pixelRatio = window.devicePixelRatio > 1 ? 2 : 1
      const canvas = document.createElement("canvas")
      canvas.width = Math.round(width * pixelRatio)
      canvas.height = Math.round(height * pixelRatio)

      const context = canvas.getContext("2d")
      if (!context) {
        throw new Error("Canvas context unavailable")
      }

      context.scale(pixelRatio, pixelRatio)
      context.fillStyle = "#ffffff"
      context.fillRect(0, 0, width, height)

      const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" })
      const blobUrl = URL.createObjectURL(blob)
      const image = new Image()

      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve()
        image.onerror = () => reject(new Error("Could not load SVG into image"))
        image.src = blobUrl
      })

      context.drawImage(image, 0, 0, width, height)
      URL.revokeObjectURL(blobUrl)

      const pngBlob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, "image/png")
      })

      if (!pngBlob) {
        throw new Error("PNG export failed")
      }

      downloadBlob(pngBlob, `visual-architect-${diagramId}.png`)
      showStatus("Exported PNG")
    } catch (error) {
      console.error("[MermaidDiagram] Failed to export PNG:", error)
      showStatus("PNG export failed")
    }
  }

  useEffect(() => {
    let cancelled = false

    async function renderDiagram() {
      try {
        ensureMermaidInitialized()
        const { svg } = await mermaid.render(`mermaid-${diagramId}`, code)

        if (!cancelled) {
          setSvg(svg)
          setError(null)
          setZoom(1)
        }
      } catch (error) {
        console.error("[MermaidDiagram] Failed to render diagram:", error)

        if (!cancelled) {
          setSvg("")
          setError(error instanceof Error ? error.message : "Invalid Mermaid syntax")
        }
      }
    }

    void renderDiagram()

    return () => {
      cancelled = true
    }
  }, [code, diagramId])

  useEffect(() => {
    return () => {
      if (statusTimerRef.current) {
        clearTimeout(statusTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!isFullscreen) {
      return
    }

    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = "hidden"

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsFullscreen(false)
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [isFullscreen])

  function renderDiagramBody(containerClassName: string) {
    if (error) {
      return (
        <div className="plasmo-p-3">
          <div className="plasmo-rounded-lg plasmo-border plasmo-border-red-200 plasmo-bg-red-50 plasmo-p-3 plasmo-text-xs plasmo-text-red-700">
            Could not render Mermaid diagram: {error}
          </div>
        </div>
      )
    }

    return (
      <div className={containerClassName}>
        <div
          className="plasmo-min-w-max [&_svg]:plasmo-h-auto [&_svg]:plasmo-max-w-none"
          style={{ zoom }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </div>
    )
  }

  function renderToolbar(isOverlay = false) {
    return (
      <div className="plasmo-flex plasmo-flex-wrap plasmo-items-center plasmo-justify-end plasmo-gap-1.5">
        {statusMessage && (
          <span className="plasmo-rounded-full plasmo-bg-sky-100 plasmo-px-2 plasmo-py-1 plasmo-text-[11px] plasmo-font-medium plasmo-text-sky-700">
            {statusMessage}
          </span>
        )}
        <button
          onClick={() => setZoom((prev) => Math.max(0.6, +(prev - 0.15).toFixed(2)))}
          className="plasmo-rounded-md plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-px-2 plasmo-py-1 plasmo-text-xs plasmo-font-medium plasmo-text-sky-700 hover:plasmo-bg-sky-100"
        >
          -
        </button>
        <button
          onClick={() => setZoom(1)}
          className="plasmo-rounded-md plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-px-2.5 plasmo-py-1 plasmo-text-xs plasmo-font-medium plasmo-text-sky-700 hover:plasmo-bg-sky-100"
        >
          {Math.round(zoom * 100)}%
        </button>
        <button
          onClick={() => setZoom((prev) => Math.min(2.25, +(prev + 0.15).toFixed(2)))}
          className="plasmo-rounded-md plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-px-2 plasmo-py-1 plasmo-text-xs plasmo-font-medium plasmo-text-sky-700 hover:plasmo-bg-sky-100"
        >
          +
        </button>
        <button
          onClick={() => void handleCopyCode()}
          className="plasmo-rounded-md plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-px-2.5 plasmo-py-1 plasmo-text-xs plasmo-font-medium plasmo-text-sky-700 hover:plasmo-bg-sky-100"
        >
          Copy Mermaid
        </button>
        <button
          onClick={handleExportSvg}
          className="plasmo-rounded-md plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-px-2.5 plasmo-py-1 plasmo-text-xs plasmo-font-medium plasmo-text-sky-700 hover:plasmo-bg-sky-100"
        >
          SVG
        </button>
        <button
          onClick={() => void handleExportPng()}
          className="plasmo-rounded-md plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-px-2.5 plasmo-py-1 plasmo-text-xs plasmo-font-medium plasmo-text-sky-700 hover:plasmo-bg-sky-100"
        >
          PNG
        </button>
        <button
          onClick={() => setShowCode((prev) => !prev)}
          className="plasmo-rounded-md plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-px-2.5 plasmo-py-1 plasmo-text-xs plasmo-font-medium plasmo-text-sky-700 hover:plasmo-bg-sky-100"
        >
          {showCode ? "Hide Code" : "View Code"}
        </button>
        <button
          onClick={() => setIsFullscreen((prev) => !prev)}
          className="plasmo-rounded-md plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-px-2.5 plasmo-py-1 plasmo-text-xs plasmo-font-medium plasmo-text-sky-700 hover:plasmo-bg-sky-100"
        >
          {isOverlay ? "Close Fullscreen" : "Fullscreen"}
        </button>
      </div>
    )
  }

  return (
    <>
      <div className="plasmo-my-3 plasmo-overflow-hidden plasmo-rounded-xl plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-shadow-sm">
        <div className="plasmo-flex plasmo-items-center plasmo-justify-between plasmo-gap-3 plasmo-border-b plasmo-border-sky-100 plasmo-bg-sky-50 plasmo-px-3 plasmo-py-2">
          <div>
            <div className="plasmo-text-xs plasmo-font-semibold plasmo-uppercase plasmo-tracking-[0.12em] plasmo-text-sky-700">
              Visual Architect
            </div>
            <div className="plasmo-text-xs plasmo-text-sky-900">Mermaid diagram</div>
          </div>
          {renderToolbar()}
        </div>

        {renderDiagramBody("plasmo-overflow-x-auto plasmo-p-3")}

        {showCode && (
          <div className="plasmo-border-t plasmo-border-sky-100 plasmo-bg-slate-950 plasmo-p-3">
            <pre className="plasmo-overflow-x-auto plasmo-text-xs plasmo-leading-relaxed plasmo-text-sky-100">
              <code>{code}</code>
            </pre>
          </div>
        )}
      </div>

      {isFullscreen && (
        <div className="plasmo-fixed plasmo-inset-0 plasmo-z-[10000] plasmo-flex plasmo-items-stretch plasmo-justify-center plasmo-bg-slate-950/70 plasmo-p-3">
          <div
            className="plasmo-absolute plasmo-inset-0"
            onClick={() => setIsFullscreen(false)}
          />
          <div className="plasmo-relative plasmo-flex plasmo-h-full plasmo-w-full plasmo-max-w-6xl plasmo-flex-col plasmo-overflow-hidden plasmo-rounded-2xl plasmo-border plasmo-border-sky-200 plasmo-bg-white plasmo-shadow-2xl">
            <div className="plasmo-flex plasmo-items-center plasmo-justify-between plasmo-gap-3 plasmo-border-b plasmo-border-sky-100 plasmo-bg-sky-50 plasmo-px-4 plasmo-py-3">
              <div>
                <div className="plasmo-text-xs plasmo-font-semibold plasmo-uppercase plasmo-tracking-[0.12em] plasmo-text-sky-700">
                  Visual Architect
                </div>
                <div className="plasmo-text-sm plasmo-font-medium plasmo-text-sky-950">
                  Fullscreen Mermaid viewer
                </div>
              </div>
              {renderToolbar(true)}
            </div>

            <div className="plasmo-flex-1 plasmo-overflow-auto plasmo-bg-slate-50">
              {renderDiagramBody("plasmo-min-h-full plasmo-overflow-auto plasmo-p-5")}
            </div>

            {showCode && (
              <div className="plasmo-max-h-[35vh] plasmo-overflow-auto plasmo-border-t plasmo-border-sky-100 plasmo-bg-slate-950 plasmo-p-4">
                <pre className="plasmo-overflow-x-auto plasmo-text-xs plasmo-leading-relaxed plasmo-text-sky-100">
                  <code>{code}</code>
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}
