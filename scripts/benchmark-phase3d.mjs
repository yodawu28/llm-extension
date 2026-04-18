#!/usr/bin/env node

const DEFAULT_BASE_URL = process.env.BENCHMARK_API_BASE_URL || "http://127.0.0.1:8000"
const DEFAULT_RUNS = Number.parseInt(process.env.BENCHMARK_RUNS || "1", 10)

const samplePages = [
  {
    title: "LTI Domain Model Overview",
    url: "http://127.0.0.1:4173/lti-domain-model-overview.html",
    source_type: "generic",
    markdown: `
# LTI Domain Model Overview

The LTI Domain Model explains how a platform integrates with an external tool. The main entities are platform, tool, registration, deployment, context, and user launch.

- A single tool registration can support multiple deployments.
- A deployment may be course-specific or institution-wide.
- The deployment determines where a launch is valid.

During launch validation, the platform sends a launch request, the tool verifies the registration, confirms deployment_id is known, checks that the launch context is in scope, and then creates a launch session.
    `.trim()
  },
  {
    title: "LTI Tool Deployment Spec",
    url: "http://127.0.0.1:4173/lti-tool-deployment-spec.html",
    source_type: "generic",
    markdown: `
# LTI Tool Deployment Spec

A client_id identifies the registration security contract. A deployment_id identifies where the tool is deployed inside the platform.

Validation sequence:
1. Resolve platform issuer and client_id
2. Load deployment by deployment_id
3. Confirm deployment belongs to the registration
4. Confirm target context is in scope
5. Create a launch session
6. Emit an audit event

Reliability defaults:
- Validation timeout: 30 seconds
- Retry policy: 2 retries with exponential backoff
    `.trim()
  },
  {
    title: "LTI QA Risk Notes",
    url: "http://127.0.0.1:4173/lti-qa-risk-notes.html",
    source_type: "generic",
    markdown: `
# LTI QA Risk Notes

Missing cases:
- deleting a deployment while old launch links still exist
- stale deployment reference after course copy
- valid client_id with mismatched deployment_id

High risk: if the tool creates a session before deployment scope is checked, the system may issue a session for an unauthorized context.
    `.trim()
  },
  {
    title: "LTI Ops Runbook",
    url: "http://127.0.0.1:4173/lti-ops-runbook.html",
    source_type: "generic",
    markdown: `
# LTI Ops Runbook

Operational defaults:
- Gateway timeout for launch validation: 45 seconds
- Retry policy: 3 retries with capped backoff
- Audit success events can be buffered
    `.trim()
  }
]

const cases = [
  {
    label: "summary",
    path: "/api/rag-summarize",
    body: {
      pages: samplePages,
      user_question: "Explain the LTI Domain Model and deployment validation flow"
    }
  },
  {
    label: "critique",
    path: "/api/critique",
    body: {
      pages: samplePages,
      user_question: "Review these docs for missing requirements, edge cases, and risks"
    }
  },
  {
    label: "diagram",
    path: "/api/generate-diagram",
    body: {
      pages: samplePages,
      user_question: "Draw a sequence diagram for tool deployment validation",
      diagram_type: "sequenceDiagram"
    }
  }
]

function parseArgs(argv) {
  let baseUrl = DEFAULT_BASE_URL
  let runs = DEFAULT_RUNS

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (arg === "--base-url" && argv[i + 1]) {
      baseUrl = argv[i + 1]
      i += 1
      continue
    }
    if (arg === "--runs" && argv[i + 1]) {
      runs = Number.parseInt(argv[i + 1], 10) || DEFAULT_RUNS
      i += 1
    }
  }

  return {
    baseUrl: baseUrl.replace(/\/$/, ""),
    runs: Math.max(1, runs)
  }
}

async function runCase(baseUrl, benchmarkCase) {
  const startedAt = performance.now()
  const response = await fetch(`${baseUrl}${benchmarkCase.path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json"
    },
    body: JSON.stringify(benchmarkCase.body)
  })
  const clientLatencyMs = Math.round(performance.now() - startedAt)
  const payload = await response.json()

  if (!response.ok) {
    throw new Error(`${benchmarkCase.label} failed: ${payload?.detail || response.statusText}`)
  }

  const usage = payload.token_usage || {}

  return {
    label: benchmarkCase.label,
    clientLatencyMs,
    serverEndToEndMs: usage.end_to_end_latency_ms ?? null,
    retrievalLatencyMs: usage.retrieval_latency_ms ?? null,
    generationLatencyMs: usage.generation_latency_ms ?? null,
    ttftMs: usage.ttft_ms ?? null,
    providerMode: usage.provider_mode || usage.provider_used || "unknown",
    provider: usage.provider_used || "unknown",
    model: usage.model_used || payload.model_used || "unknown",
    budgetPolicy: usage.context_budget_policy || "n/a",
    timingSource: usage.timing_source || "wall-clock",
    hybridPrefilter: usage.hybrid_prefilter_applied
      ? `applied:${usage.hybrid_prefilter_output_chunks ?? "?"}/${usage.hybrid_prefilter_input_chunks ?? "?"}`
      : usage.hybrid_prefilter_fallback_used
        ? "fallback"
        : "off"
  }
}

function mean(values) {
  if (!values.length) return null
  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length)
}

function printSummary(results) {
  const grouped = new Map()

  for (const result of results) {
    const bucket = grouped.get(result.label) || []
    bucket.push(result)
    grouped.set(result.label, bucket)
  }

  console.log("")
  console.log("Phase 3D benchmark summary")
  console.log("")

  for (const [label, bucket] of grouped.entries()) {
    const latest = bucket[bucket.length - 1]
    console.log(`${label}`)
    console.log(`  provider mode: ${latest.providerMode}`)
    console.log(`  provider: ${latest.provider}`)
    console.log(`  model: ${latest.model}`)
    console.log(`  budget policy: ${latest.budgetPolicy}`)
    console.log(`  hybrid prefilter: ${latest.hybridPrefilter}`)
    console.log(`  timing source: ${latest.timingSource}`)
    console.log(`  client latency avg: ${mean(bucket.map((item) => item.clientLatencyMs)) ?? "n/a"} ms`)
    console.log(`  server end-to-end avg: ${mean(bucket.map((item) => item.serverEndToEndMs).filter(Boolean)) ?? "n/a"} ms`)
    console.log(`  retrieval avg: ${mean(bucket.map((item) => item.retrievalLatencyMs).filter(Boolean)) ?? "n/a"} ms`)
    console.log(`  generation avg: ${mean(bucket.map((item) => item.generationLatencyMs).filter(Boolean)) ?? "n/a"} ms`)
    console.log(`  ttft avg: ${mean(bucket.map((item) => item.ttftMs).filter(Boolean)) ?? "n/a"} ms`)
    console.log("")
  }
}

async function main() {
  const { baseUrl, runs } = parseArgs(process.argv.slice(2))
  const results = []

  console.log(`Benchmarking ${baseUrl} with ${runs} run(s) per case`)

  for (let runIndex = 0; runIndex < runs; runIndex += 1) {
    for (const benchmarkCase of cases) {
      const result = await runCase(baseUrl, benchmarkCase)
      results.push(result)
      console.log(
        `[run ${runIndex + 1}] ${result.label}: client=${result.clientLatencyMs}ms server=${result.serverEndToEndMs ?? "n/a"}ms retrieval=${result.retrievalLatencyMs ?? "n/a"}ms generation=${result.generationLatencyMs ?? "n/a"}ms ttft=${result.ttftMs ?? "n/a"}ms`
      )
    }
  }

  printSummary(results)
}

main().catch((error) => {
  console.error(error.message || error)
  process.exit(1)
})
