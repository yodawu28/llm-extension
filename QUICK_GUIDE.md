# Quick Guide: Local LLM Setup

File nay huong dan nhanh cho user noi bo muon chay `Web Context Assistant` voi local LLM tren Mac.

## Chon mode

Co 2 mode thuc dung:

1. `Local-only`
- Chat model chay bang Ollama
- Embeddings chay local
- Khong can cloud API key
- Phu hop de PoC, demo local, va test chi phi `0`

2. `Hybrid`
- Local Ollama chi dung de prefilter/rut gon context
- Cloud model tra loi cuoi
- Phu hop khi muon chat luong cloud nhung giam token burn

## Prerequisites

Can co:
- macOS
- `pnpm`
- Python 3.10+
- Ollama da cai tren may

Model khuyen dung:
- `gemma2:2b`
- `deepseek-coder-v2:16b-lite-instruct`

Pull model:

```bash
ollama pull gemma2:2b
ollama pull deepseek-coder-v2:16b-lite-instruct
```

Neu Ollama chua tu chay, start no tren may host.

## Extension setup

Tu repo root:

```bash
pnpm install
cp .env.example .env.local
```

Mac dinh:

```env
PLASMO_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Backend setup

```bash
cd backend
cp .env.example .env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

## Option A: Local-only

Sua [backend/.env](/Users/long.vo/workspaces/personal/side-projects/llm-extensions/backend/.env):

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL_SUMMARY=gemma2:2b
OLLAMA_MODEL_REASONING=deepseek-coder-v2:16b-lite-instruct
OLLAMA_REQUEST_TIMEOUT_SECONDS=45

EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_DIMENSIONS=384

REQUEST_TIMEOUT_SECONDS=120
LOG_LEVEL=INFO
```

Ghi chu:
- neu backend chay trong Docker, doi `OLLAMA_BASE_URL` thanh `http://host.docker.internal:11434`
- summary thuong dung `OLLAMA_MODEL_SUMMARY`
- critique/diagram thuong dung `OLLAMA_MODEL_REASONING`

## Option B: Hybrid

Sua [backend/.env](/Users/long.vo/workspaces/personal/side-projects/llm-extensions/backend/.env):

```env
LLM_PROVIDER=hybrid
HYBRID_CLOUD_PROVIDER=openai

AWS_GATEWAY_URL=https://genai-gateway.flava-cloud.com/v1
PAT_TOKEN=your-internal-pat
OPENAI_MODEL_SUMMARY=gpt-4o-mini
OPENAI_MODEL_REASONING=o4-mini

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL_SUMMARY=gemma2:2b
HYBRID_PREFILTER_ENABLED=true
HYBRID_PREFILTER_MAX_CHUNKS=4
HYBRID_PREFILTER_TIMEOUT_SECONDS=20

EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_DIMENSIONS=384

REQUEST_TIMEOUT_SECONDS=120
LOG_LEVEL=INFO
```

Hybrid mode hoat dong nhu sau:
- retrieval van chay trong backend
- local Ollama cat gon retrieved chunks
- cloud model nhan context da rut gon va tra loi cuoi
- neu local prefilter fail, backend fallback ve full retrieved context

## Validate env

Truoc khi start backend:

```bash
node scripts/check-env.mjs backend/.env
```

Neu pass, ban se thay summary config va dong `Environment validation passed.`

## Run backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Quick health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## Run extension

```bash
pnpm dev
```

Hoac build prod:

```bash
pnpm build
```

Sau do:
1. mo `chrome://extensions`
2. bat `Developer mode`
3. `Load unpacked`
4. chon [build/chrome-mv3-prod](/Users/long.vo/workspaces/personal/side-projects/llm-extensions/build/chrome-mv3-prod) neu build prod

## Recommended first test

1. mo 1-3 page
2. pin page vao context basket
3. hoi:

```text
summarize only this page
```

hoac:

```text
critique this page only
```

4. mo `Technical Diagnostics`

## What to check in Diagnostics

### Local-only

Ky vong thay:
- `Provider: ollama`
- `Model`
- `Budget policy`
- `Retrieval time`
- `Generation time`
- `TTFT` neu co

### Hybrid

Ky vong thay:
- `Provider mode: hybrid`
- `Provider: openai` hoac `anthropic`
- `Hybrid prefilter: applied` hoac `fallback`
- `Prefilter model`
- `Prefilter chunks`
- `Prefilter time`

## Troubleshooting

### Backend khong len

Check:

```bash
node scripts/check-env.mjs backend/.env
```

### Ollama khong duoc goi

Check:
- `OLLAMA_BASE_URL` dung host
- Ollama dang chay
- model da duoc `pull`

### Embeddings bi fail

Neu gateway chan embeddings, dung:

```env
EMBEDDING_PROVIDER=local
```

### Hybrid khong co prefilter

Check:
- `LLM_PROVIDER=hybrid`
- `HYBRID_PREFILTER_ENABLED=true`
- `OLLAMA_MODEL_SUMMARY` khong rong
- diagnostics co `Hybrid prefilter`

## Useful commands

```bash
node scripts/check-env.mjs backend/.env
pnpm benchmark:phase3d
pnpm benchmark:phase3d -- --base-url http://127.0.0.1:8000 --runs 3
```
