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
    ollama: "ollama",
    local: "ollama",
    "local-llm": "ollama",
    local_llm: "ollama",
    hybrid: "hybrid",
    "hybrid-local-cloud": "hybrid",
    hybrid_local_cloud: "hybrid",
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
  const openAiSummaryModel = String(env.OPENAI_MODEL_SUMMARY || "").trim()
  const openAiReasoningModel = String(env.OPENAI_MODEL_REASONING || "").trim()
  const effectiveOpenAiModel = openAiSummaryModel || openAiModel || openAiReasoningModel
  const anthropicModel = String(env.ANTHROPIC_MODEL || "").trim()
  const anthropicSummaryModel = String(env.ANTHROPIC_MODEL_SUMMARY || "").trim()
  const anthropicReasoningModel = String(env.ANTHROPIC_MODEL_REASONING || "").trim()
  const effectiveAnthropicModel = anthropicSummaryModel || anthropicModel || anthropicReasoningModel
  const ollamaBaseUrl = sanitizeBaseUrl(env.OLLAMA_BASE_URL || "http://host.docker.internal:11434")
  const ollamaSummaryModel = String(env.OLLAMA_MODEL_SUMMARY || env.OLLAMA_MODEL || "").trim()
  const ollamaReasoningModel = String(env.OLLAMA_MODEL_REASONING || "").trim()
  const ollamaTimeoutSeconds = parsePositiveInteger(env.OLLAMA_REQUEST_TIMEOUT_SECONDS || "45")
  const hybridCloudProvider = normalizeProvider(env.HYBRID_CLOUD_PROVIDER || "openai")
  const hybridPrefilterTimeoutSeconds = parsePositiveInteger(env.HYBRID_PREFILTER_TIMEOUT_SECONDS || "20")
  const hybridPrefilterMaxChunks = parsePositiveInteger(env.HYBRID_PREFILTER_MAX_CHUNKS || "4")
  const hasPatToken = Boolean(String(env.PAT_TOKEN || env.OPENAI_PAT || "").trim())
  const hasOpenAiKey = Boolean(String(env.OPENAI_API_KEY || "").trim())
  const hasAnthropicKey = Boolean(String(env.ANTHROPIC_API_KEY || "").trim())
  const timeoutSeconds = parsePositiveInteger(env.REQUEST_TIMEOUT_SECONDS || "120")
  const logLevel = String(env.LOG_LEVEL || "INFO").trim().toUpperCase()

  if (llmProvider === "openai") {
    if (!hasPatToken && !hasOpenAiKey) {
      errors.push("OpenAI-compatible provider requires `PAT_TOKEN` or `OPENAI_API_KEY`.")
    }

    if (!effectiveOpenAiModel) {
      errors.push("OpenAI-compatible provider requires `OPENAI_MODEL`, `MODEL`, or `OPENAI_MODEL_SUMMARY`.")
    }

    if (hasPatToken && !openAiBaseUrl) {
      warnings.push("`PAT_TOKEN` is set but `AWS_GATEWAY_URL`/`OPENAI_BASE_URL` is empty.")
    }
  } else if (llmProvider === "anthropic") {
    if (!hasAnthropicKey) {
      errors.push("Anthropic provider requires `ANTHROPIC_API_KEY`.")
    }

    if (!effectiveAnthropicModel) {
      errors.push("Anthropic provider requires `ANTHROPIC_MODEL` or `ANTHROPIC_MODEL_SUMMARY`.")
    }
  } else if (llmProvider === "ollama") {
    if (!ollamaBaseUrl) {
      errors.push("Ollama provider requires `OLLAMA_BASE_URL`.")
    }

    if (!ollamaSummaryModel) {
      errors.push("Ollama provider requires `OLLAMA_MODEL_SUMMARY` or `OLLAMA_MODEL`.")
    }

    if (!ollamaReasoningModel) {
      warnings.push("`OLLAMA_MODEL_REASONING` is empty. Critique/diagram will fall back to the summary model.")
    }

    if (!ollamaTimeoutSeconds) {
      errors.push("`OLLAMA_REQUEST_TIMEOUT_SECONDS` must be a positive integer.")
    }
  } else if (llmProvider === "hybrid") {
    if (!["openai", "anthropic"].includes(hybridCloudProvider)) {
      errors.push("Hybrid mode requires `HYBRID_CLOUD_PROVIDER=openai` or `anthropic`.")
    }

    if (hybridCloudProvider === "openai") {
      if (!hasPatToken && !hasOpenAiKey) {
        errors.push("Hybrid mode with OpenAI-compatible cloud provider requires `PAT_TOKEN` or `OPENAI_API_KEY`.")
      }
      if (!effectiveOpenAiModel) {
        errors.push("Hybrid mode with OpenAI-compatible cloud provider requires `OPENAI_MODEL` or `OPENAI_MODEL_SUMMARY`.")
      }
      if (hasPatToken && !openAiBaseUrl) {
        warnings.push("Hybrid mode with PAT token usually needs `AWS_GATEWAY_URL` or `OPENAI_BASE_URL`.")
      }
    } else if (hybridCloudProvider === "anthropic") {
      if (!hasAnthropicKey) {
        errors.push("Hybrid mode with Anthropic cloud provider requires `ANTHROPIC_API_KEY`.")
      }
      if (!effectiveAnthropicModel) {
        errors.push("Hybrid mode with Anthropic cloud provider requires `ANTHROPIC_MODEL` or `ANTHROPIC_MODEL_SUMMARY`.")
      }
    }

    if (!ollamaBaseUrl) {
      errors.push("Hybrid mode requires `OLLAMA_BASE_URL` for local prefilter.")
    }
    if (!ollamaSummaryModel) {
      errors.push("Hybrid mode requires `OLLAMA_MODEL_SUMMARY` or `OLLAMA_MODEL` for local prefilter.")
    }
    if (!hybridPrefilterTimeoutSeconds) {
      errors.push("`HYBRID_PREFILTER_TIMEOUT_SECONDS` must be a positive integer.")
    }
    if (!hybridPrefilterMaxChunks) {
      errors.push("`HYBRID_PREFILTER_MAX_CHUNKS` must be a positive integer.")
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
  console.log(`- Base URL: ${llmProvider === "ollama" ? ollamaBaseUrl : llmProvider === "hybrid" ? (hybridCloudProvider === "openai" ? openAiBaseUrl || "(default/openai)" : "(anthropic)") : openAiBaseUrl || "(default/openai)"}`)
  console.log(`- Chat model: ${llmProvider === "ollama" ? ollamaSummaryModel || "(missing)" : llmProvider === "anthropic" ? effectiveAnthropicModel || "(missing)" : llmProvider === "hybrid" ? (hybridCloudProvider === "anthropic" ? effectiveAnthropicModel || "(missing)" : effectiveOpenAiModel || "(missing)") : effectiveOpenAiModel || "(missing)"}`)
  if (llmProvider === "openai" && openAiReasoningModel) {
    console.log(`- Reasoning model: ${openAiReasoningModel}`)
  }
  if (llmProvider === "anthropic" && anthropicReasoningModel) {
    console.log(`- Reasoning model: ${anthropicReasoningModel}`)
  }
  if (llmProvider === "ollama") {
    console.log(`- Reasoning model: ${ollamaReasoningModel || ollamaSummaryModel || "(missing)"}`)
    console.log(`- Ollama timeout: ${ollamaTimeoutSeconds || "(invalid)"}`)
  }
  if (llmProvider === "hybrid") {
    console.log(`- Cloud provider: ${hybridCloudProvider}`)
    console.log(`- Prefilter model: ${ollamaSummaryModel || "(missing)"}`)
    console.log(`- Prefilter timeout: ${hybridPrefilterTimeoutSeconds || "(invalid)"}`)
    console.log(`- Prefilter max chunks: ${hybridPrefilterMaxChunks || "(invalid)"}`)
  }
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
