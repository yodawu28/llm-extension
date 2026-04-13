# Web Context Assistant - Backend API

FastAPI backend for summarizing webpage content using LLMs (OpenAI GPT-4o or Anthropic Claude 3.5 Sonnet).

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

Edit `.env` and add your API keys:

```env
# Choose provider
LLM_PROVIDER=openai  # or anthropic

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=
PAT_TOKEN=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536

# Or Anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Internal gateway example:

```env
LLM_PROVIDER=openai
AWS_GATEWAY_URL=https://genai-gateway.flava-cloud.com/
PAT_TOKEN=your-internal-pat
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 4. Run Server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Server will start at: **http://localhost:8000**

API Documentation: **http://localhost:8000/docs**

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
