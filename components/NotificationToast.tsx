/**
 * Notification Toast Component - Show proactive suggestions
 *
 * Displays discovered linked resources with actions to add to context.
 */

import { useState, useEffect } from "react"
import type { LinkedResource } from "~lib/link-scanner"
import type { FetchProgress } from "~lib/background-fetcher"

export interface ToastNotification {
  id: string
  title: string
  message: string
  icon?: string
  resources?: LinkedResource[]
  progress?: FetchProgress
  duration?: number  // Auto-dismiss after ms (0 = manual dismiss)
  onAction?: (action: string) => void
  onDismiss?: () => void
}

interface NotificationToastProps {
  notification: ToastNotification
  onClose: () => void
}

export function NotificationToast({ notification, onClose }: NotificationToastProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isExiting, setIsExiting] = useState(false)

  useEffect(() => {
    // Slide in animation
    setTimeout(() => setIsVisible(true), 10)

    // Auto-dismiss if duration set
    if (notification.duration && notification.duration > 0) {
      const timer = setTimeout(() => {
        handleDismiss()
      }, notification.duration)

      return () => clearTimeout(timer)
    }
  }, [notification.duration])

  const handleDismiss = () => {
    setIsExiting(true)
    setTimeout(() => {
      notification.onDismiss?.()
      onClose()
    }, 300)
  }

  const handleAction = (action: string) => {
    notification.onAction?.(action)
    handleDismiss()
  }

  return (
    <div
      className={`plasmo-fixed plasmo-bottom-4 plasmo-right-4 plasmo-z-[9999] plasmo-transition-all plasmo-duration-300 ${
        isVisible && !isExiting
          ? "plasmo-translate-y-0 plasmo-opacity-100"
          : "plasmo-translate-y-4 plasmo-opacity-0"
      }`}
    >
      <div className="plasmo-bg-white plasmo-rounded-lg plasmo-shadow-2xl plasmo-border plasmo-border-gray-200 plasmo-max-w-md plasmo-overflow-hidden">
        {/* Header */}
        <div className="plasmo-flex plasmo-items-center plasmo-gap-3 plasmo-p-4 plasmo-bg-gradient-to-r plasmo-from-blue-500 plasmo-to-indigo-500">
          <div className="plasmo-text-3xl">{notification.icon || "💡"}</div>
          <div className="plasmo-flex-1">
            <h3 className="plasmo-text-white plasmo-font-bold plasmo-text-base">
              {notification.title}
            </h3>
            <p className="plasmo-text-blue-100 plasmo-text-sm plasmo-mt-0.5">
              {notification.message}
            </p>
          </div>
          <button
            onClick={handleDismiss}
            className="plasmo-text-white hover:plasmo-bg-white hover:plasmo-bg-opacity-20 plasmo-rounded plasmo-p-1 plasmo-transition-colors"
          >
            <svg
              className="plasmo-w-5 plasmo-h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Resources List */}
        {notification.resources && notification.resources.length > 0 && (
          <div className="plasmo-p-4 plasmo-bg-gray-50 plasmo-max-h-60 plasmo-overflow-y-auto">
            <div className="plasmo-space-y-2">
              {notification.resources.map((resource, idx) => (
                <div
                  key={idx}
                  className="plasmo-bg-white plasmo-rounded plasmo-p-3 plasmo-border plasmo-border-gray-200"
                >
                  <div className="plasmo-flex plasmo-items-start plasmo-gap-2">
                    <span className="plasmo-text-lg plasmo-flex-shrink-0">
                      {getResourceIcon(resource.type)}
                    </span>
                    <div className="plasmo-flex-1 plasmo-min-w-0">
                      <div className="plasmo-text-sm plasmo-font-semibold plasmo-text-gray-800 plasmo-truncate">
                        {resource.title}
                      </div>
                      <div className="plasmo-text-xs plasmo-text-gray-500 plasmo-mt-0.5 plasmo-line-clamp-2">
                        {resource.context}
                      </div>
                      <div className="plasmo-flex plasmo-items-center plasmo-gap-2 plasmo-mt-1">
                        <span
                          className={`plasmo-text-xs plasmo-px-2 plasmo-py-0.5 plasmo-rounded-full ${getConfidenceBadgeClass(
                            resource.confidence
                          )}`}
                        >
                          {resource.confidence}
                        </span>
                        <span className="plasmo-text-xs plasmo-text-gray-400">
                          {resource.type}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Progress Bar */}
        {notification.progress && notification.progress.status === "fetching" && (
          <div className="plasmo-px-4 plasmo-py-3 plasmo-bg-gray-50 plasmo-border-t plasmo-border-gray-200">
            <div className="plasmo-flex plasmo-items-center plasmo-justify-between plasmo-mb-2">
              <span className="plasmo-text-xs plasmo-text-gray-600">
                {notification.progress.current || "Fetching..."}
              </span>
              <span className="plasmo-text-xs plasmo-font-semibold plasmo-text-gray-800">
                {notification.progress.completed} / {notification.progress.total}
              </span>
            </div>
            <div className="plasmo-w-full plasmo-bg-gray-200 plasmo-rounded-full plasmo-h-2">
              <div
                className="plasmo-bg-blue-500 plasmo-h-2 plasmo-rounded-full plasmo-transition-all plasmo-duration-300"
                style={{
                  width: `${
                    (notification.progress.completed / notification.progress.total) * 100
                  }%`
                }}
              />
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="plasmo-flex plasmo-gap-2 plasmo-p-4 plasmo-bg-white plasmo-border-t plasmo-border-gray-200">
          <button
            onClick={() => handleAction("add-all")}
            disabled={notification.progress?.status === "fetching"}
            className="plasmo-flex-1 plasmo-px-4 plasmo-py-2 plasmo-bg-blue-500 plasmo-text-white plasmo-rounded-lg plasmo-text-sm plasmo-font-medium hover:plasmo-bg-blue-600 plasmo-transition-colors disabled:plasmo-opacity-50 disabled:plasmo-cursor-not-allowed"
          >
            {notification.progress?.status === "fetching" ? "Fetching..." : "Add All"}
          </button>
          <button
            onClick={() => handleAction("pick")}
            disabled={notification.progress?.status === "fetching"}
            className="plasmo-px-4 plasmo-py-2 plasmo-bg-gray-100 plasmo-text-gray-700 plasmo-rounded-lg plasmo-text-sm plasmo-font-medium hover:plasmo-bg-gray-200 plasmo-transition-colors disabled:plasmo-opacity-50 disabled:plasmo-cursor-not-allowed"
          >
            Pick
          </button>
          <button
            onClick={handleDismiss}
            className="plasmo-px-4 plasmo-py-2 plasmo-text-gray-500 plasmo-rounded-lg plasmo-text-sm plasmo-font-medium hover:plasmo-bg-gray-100 plasmo-transition-colors"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  )
}

function getResourceIcon(type: string): string {
  const icons = {
    confluence: "📄",
    jira: "🎫",
    github: "🐙",
    doc: "📝",
    generic: "🔗"
  }
  return icons[type as keyof typeof icons] || "🔗"
}

function getConfidenceBadgeClass(confidence: string): string {
  const classes = {
    high: "plasmo-bg-green-100 plasmo-text-green-800",
    medium: "plasmo-bg-yellow-100 plasmo-text-yellow-800",
    low: "plasmo-bg-gray-100 plasmo-text-gray-600"
  }
  return classes[confidence as keyof typeof classes] || classes.low
}
