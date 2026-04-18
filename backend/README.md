# Web Context Assistant - Backend API

FastAPI backend for summarizing webpage content using OpenAI-compatible gateways, Anthropic, local Ollama models, or a hybrid local-prefilter plus cloud-answer mode.

## Prerequisites

- Python 3.10+
- OpenAI API key OR Anthropic API key

## Setup

### 1. Create Virtual Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and choose one provider flow:

```env
# Choose provider
LLM_PROVIDER=openai  # or anthropic, ollama, hybrid

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_SUMMARY=
OPENAI_MODEL_REASONING=
OPENAI_BASE_URL=
PAT_TOKEN=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536

# Or Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL_SUMMARY=
ANTHROPIC_MODEL_REASONING=
```

Internal gateway example:

```env
LLM_PROVIDER=openai
AWS_GATEWAY_URL=https://genai-gateway.flava-cloud.com/
PAT_TOKEN=your-internal-pat
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
REQUEST_TIMEOUT_SECONDS=120
```

Local Ollama example:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL_SUMMARY=gemma2:2b
OLLAMA_MODEL_REASONING=deepseek-coder-v2:16b-lite-instruct
OLLAMA_REQUEST_TIMEOUT_SECONDS=45
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_DIMENSIONS=384
```

Notes for Ollama:
- Run Ollama natively on the Mac host, not inside the backend container
- If the backend runs in Docker, change `OLLAMA_BASE_URL` to `http://host.docker.internal:11434`
- Summary-style tasks use `OLLAMA_MODEL_SUMMARY`
- Critique and diagram tasks use `OLLAMA_MODEL_REASONING`
- Local provider cost is reported as `$0.0000` in diagnostics

Hybrid example:

```env
LLM_PROVIDER=hybrid
HYBRID_CLOUD_PROVIDER=openai
AWS_GATEWAY_URL=https://genai-gateway.flava-cloud.com/v1
PAT_TOKEN=your-internal-pat
OPENAI_MODEL_SUMMARY=gpt-4o-mini
OPENAI_MODEL_REASONING=o4-mini
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL_SUMMARY=gemma2:2b
HYBRID_PREFILTER_MAX_CHUNKS=4
HYBRID_PREFILTER_TIMEOUT_SECONDS=20
EMBEDDING_PROVIDER=local
```

Notes for hybrid mode:
- Retrieval still runs in the backend pipeline as usual
- Ollama is used only to shrink the retrieved chunk set before the final cloud answer
- If local prefilter fails or times out, the backend falls back to the full retrieved context
- Diagnostics show whether hybrid prefilter was applied, how many chunks were kept, and how long it took

Optional task routing for hosted providers:

```env
# OpenAI-compatible
OPENAI_MODEL_SUMMARY=gpt-4o-mini
OPENAI_MODEL_REASONING=o4-mini

# Anthropic
ANTHROPIC_MODEL_SUMMARY=claude-3-5-haiku-20241022
ANTHROPIC_MODEL_REASONING=claude-3-5-sonnet-20241022
```

If these are empty, the backend reuses the default provider model for all tasks.

### 4. Run Server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Server will start at: **http://localhost:8000**

API Documentation: **http://localhost:8000/docs**

## Docker

Build and run the backend from the repo root:

```bash
cp ../.env.compose.example ../.env.compose
cp .env.example .env
docker compose --env-file ../.env.compose up --build -d backend
```

This uses [backend/Dockerfile](/Users/long.vo/workspaces/personal/side-projects/llm-extensions/backend/Dockerfile) and loads environment variables from `backend/.env`.

Notes:
- The container exposes port `8000`
- The image installs both base backend dependencies and optional local embedding/reranker dependencies
- `EMBEDDING_PROVIDER=local` is supported inside Docker too
- `LLM_PROVIDER=ollama` is supported too, using `OLLAMA_BASE_URL=http://host.docker.internal:11434`
- The compose stack includes a backend healthcheck and an optional smoke-check service

Smoke check:

```bash
docker compose --env-file ../.env.compose --profile check up backend-smoke
```

Validate env before startup:

```bash
node ../scripts/check-env.mjs .env
```

## API Endpoints

### `GET /api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "app_name": "Web Context Assistant API",
  "llm_provider": "openai",
  "model": "gpt-4o"
}
```

### `POST /api/summarize`

Summarize multiple pages.

**Request:**
```json
{
  "pages": [
    {
      "title": "React Documentation",
      "url": "https://react.dev/",
      "markdown": "# React\n\nA JavaScript library...",
      "source_type": "generic"
    }
  ],
  "user_question": "What are the main features?"
}
```

### `POST /api/critique`

Review pinned pages for missing requirements, edge cases, and risks.

**Request:**
```json
{
  "pages": [
    {
      "title": "Authentication PRD",
      "url": "https://example.com/auth",
      "markdown": "# Authentication\n\nUsers can sign in with email...",
      "source_type": "confluence"
    }
  ],
  "user_question": "Review this PRD for missing edge cases and risky assumptions"
}
```

**Response:**
```json
{
  "summary": "The document covers the main happy path but leaves several operational gaps.",
  "issues": [
    {
      "title": "Missing offline recovery flow",
      "severity": "high",
      "category": "reliability",
      "evidence": "\"Users can sign in with email...\"",
      "risk": "The PRD does not describe what happens when the network drops mid-session.",
      "suggestion": "Define reconnect, retry, and token refresh behavior.",
      "source_title": "Authentication PRD"
    }
  ],
  "citations": [
    {
      "page_title": "Authentication PRD",
      "page_url": "https://example.com/auth",
      "source_type": "confluence"
    }
  ],
  "token_usage": {
    "input_tokens": 1234,
    "output_tokens": 456,
    "total_tokens": 1690
  },
  "model_used": "gpt-4o"
}
```

**Response:**
```json
{
  "summary": "React is a JavaScript library for building user interfaces...",
  "citations": [
    {
      "page_title": "React Documentation",
      "page_url": "https://react.dev/",
      "source_type": "generic"
    }
  ],
  "token_usage": {
    "input_tokens": 1234,
    "output_tokens": 567,
    "total_tokens": 1801
  },
  "model_used": "gpt-4o"
}
```

## Development

### Run with auto-reload

```bash
uvicorn main:app --reload
```

### Test with curl

```bash
# Health check
curl http://localhost:8000/api/health

# Summarize
curl -X POST http://localhost:8000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "pages": [
      {
        "title": "Test Page",
        "url": "https://example.com",
        "markdown": "# Test\nThis is a test.",
        "source_type": "generic"
      }
    ]
  }'
```

## Project Structure

```
backend/
├── main.py              # FastAPI app
├── config/
│   └── settings.py      # Pydantic settings
├── routers/
│   └── summarize.py     # API endpoints
├── services/
│   └── llm_service.py   # LLM integration
└── schemas/
    └── requests.py      # Request/Response models
```

## Troubleshooting

**CORS errors from extension:**
- Make sure `CORS_ORIGINS` in `.env` includes `chrome-extension://*`

**API key errors:**
- Check `.env` file has correct API key
- Verify `LLM_PROVIDER` matches the API key you provided
- If you use the internal gateway, set `PAT_TOKEN` and `AWS_GATEWAY_URL`

**Module not found:**
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`
