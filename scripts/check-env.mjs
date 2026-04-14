import fs from "node:fs"
import path from "node:path"

const envPathArg = process.argv[2] || "backend/.env"
const envPath = path.resolve(process.cwd(), envPathArg)

function readEnvFile(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Env file not found: ${filePath}`)
  }

  const content = fs.readFileSync(filePath, "utf8")
  const entries = {}

  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith("#")) continue

    const separatorIndex = line.indexOf("=")
    if (separatorIndex === -1) continue

    const key = line.slice(0, separatorIndex).trim()
    let value = line.slice(separatorIndex + 1).trim()

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1)
    }

    entries[key] = value
  }

  return entries
}

function normalizeProvider(value) {
  const normalized = String(value || "").trim().toLowerCase()
  const aliases = {
    openai: "openai",
    anthropic: "anthropic",
    "ly-chatai": "openai",
    ly_chatai: "openai",
    gateway: "openai",
    "openai-compatible": "openai",
    openai_compatible: "openai"
  }

  return aliases[normalized] || normalized || "openai"
}

function normalizeEmbeddingProvider(value) {
  const normalized = String(value || "").trim().toLowerCase()
  const aliases = {
    openai: "openai",
    gateway: "openai",
    remote: "openai",
    local: "local",
    "sentence-transformers": "local",
    sentence_transformers: "local",
    sbert: "local"
  }

  return aliases[normalized] || normalized || "openai"
}

function sanitizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "")
}

function parsePositiveInteger(value) {
  if (!String(value || "").trim()) return null
  const parsed = Number.parseInt(String(value), 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function printList(title, items) {
  if (items.length === 0) return
  console.log(`${title}:`)
  for (const item of items) {
    console.log(`- ${item}`)
  }
}

try {
  const env = readEnvFile(envPath)
  const errors = []
  const warnings = []

  const llmProvider = normalizeProvider(env.LLM_PROVIDER || env.MODEL_PROVIDER)
  const embeddingProvider = normalizeEmbeddingProvider(env.EMBEDDING_PROVIDER)
  const openAiBaseUrl = sanitizeBaseUrl(env.AWS_GATEWAY_URL || env.OPENAI_BASE_URL)
  const openAiModel = String(env.OPENAI_MODEL || env.MODEL || "").trim()
  const anthropicModel = String(env.ANTHROPIC_MODEL || "").trim()
  const hasPatToken = Boolean(String(env.PAT_TOKEN || env.OPENAI_PAT || "").trim())
  const hasOpenAiKey = Boolean(String(env.OPENAI_API_KEY || "").trim())
  const hasAnthropicKey = Boolean(String(env.ANTHROPIC_API_KEY || "").trim())
  const timeoutSeconds = parsePositiveInteger(env.REQUEST_TIMEOUT_SECONDS || "120")
  const logLevel = String(env.LOG_LEVEL || "INFO").trim().toUpperCase()

  if (llmProvider === "openai") {
    if (!hasPatToken && !hasOpenAiKey) {
      errors.push("OpenAI-compatible provider requires `PAT_TOKEN` or `OPENAI_API_KEY`.")
    }

    if (!openAiModel) {
      errors.push("OpenAI-compatible provider requires `OPENAI_MODEL` or `MODEL`.")
    }

    if (hasPatToken && !openAiBaseUrl) {
      warnings.push("`PAT_TOKEN` is set but `AWS_GATEWAY_URL`/`OPENAI_BASE_URL` is empty.")
    }
  } else if (llmProvider === "anthropic") {
    if (!hasAnthropicKey) {
      errors.push("Anthropic provider requires `ANTHROPIC_API_KEY`.")
    }

    if (!anthropicModel) {
      errors.push("Anthropic provider requires `ANTHROPIC_MODEL`.")
    }
  } else {
    errors.push(`Unsupported provider "${env.LLM_PROVIDER || env.MODEL_PROVIDER || ""}".`)
  }

  if (embeddingProvider === "openai") {
    if (!String(env.OPENAI_EMBEDDING_MODEL || "").trim()) {
      errors.push("OpenAI embedding provider requires `OPENAI_EMBEDDING_MODEL`.")
    }

    if (!parsePositiveInteger(env.OPENAI_EMBEDDING_DIMENSIONS || "1536")) {
      errors.push("`OPENAI_EMBEDDING_DIMENSIONS` must be a positive integer.")
    }

    if (llmProvider === "openai" && hasPatToken && !openAiBaseUrl) {
      warnings.push("Gateway-style embeddings usually need `AWS_GATEWAY_URL` or `OPENAI_BASE_URL`.")
    }
  } else if (embeddingProvider === "local") {
    if (!String(env.LOCAL_EMBEDDING_MODEL || "").trim()) {
      errors.push("Local embedding provider requires `LOCAL_EMBEDDING_MODEL`.")
    }

    if (!parsePositiveInteger(env.LOCAL_EMBEDDING_DIMENSIONS || "384")) {
      errors.push("`LOCAL_EMBEDDING_DIMENSIONS` must be a positive integer.")
    }
  } else {
    errors.push(`Unsupported embedding provider "${env.EMBEDDING_PROVIDER || ""}".`)
  }

  if (!timeoutSeconds) {
    errors.push("`REQUEST_TIMEOUT_SECONDS` must be a positive integer.")
  }

  if (!["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].includes(logLevel)) {
    errors.push("`LOG_LEVEL` must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")
  }

  if (!String(env.CORS_ORIGINS || "").trim()) {
    warnings.push("`CORS_ORIGINS` is empty. The Chrome extension may fail to call the backend.")
  }

  console.log(`Checking env file: ${envPath}`)
  console.log(`- LLM provider: ${llmProvider}`)
  console.log(`- Embedding provider: ${embeddingProvider}`)
  console.log(`- Base URL: ${openAiBaseUrl || "(default/openai)"}`)
  console.log(`- Chat model: ${openAiModel || anthropicModel || "(missing)"}`)
  console.log(`- Timeout: ${timeoutSeconds || "(invalid)"}`)

  printList("Warnings", warnings)

  if (errors.length > 0) {
    printList("Errors", errors)
    process.exit(1)
  }

  console.log("Environment validation passed.")
} catch (error) {
  console.error(`Environment validation failed: ${error instanceof Error ? error.message : String(error)}`)
  process.exit(1)
}
