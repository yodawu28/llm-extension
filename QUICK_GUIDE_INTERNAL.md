# Quick Guide Internal

Ban nay danh cho user noi bo muon chay nhanh `Web Context Assistant` voi local LLM tren Mac.

## Chon mode

### Mode 1: Local-only

Dung khi:
- muon chay hoan toan local
- khong can cloud key
- chap nhan toc do cham hon

### Mode 2: Hybrid

Dung khi:
- muon chat luong cloud
- muon local Ollama cat gon context truoc de giam token burn

## 1. Cai model local

Can Ollama tren Mac.

Pull model:

```bash
ollama pull gemma2:2b
ollama pull deepseek-coder-v2:16b-lite-instruct
```

## 2. Setup project

```bash
pnpm install
cp .env.example .env.local

cd backend
cp .env.example .env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

Extension env:

```env
PLASMO_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## 3. Chon config backend

### Option A: Local-only

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

### Option B: Hybrid

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

## 4. Validate config

```bash
node scripts/check-env.mjs backend/.env
```

Neu pass, se thay:

```text
Environment validation passed.
```

## 5. Run backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## 6. Run extension

```bash
pnpm build
```

Sau do:
1. mo `chrome://extensions`
2. bat `Developer mode`
3. `Load unpacked`
4. chon [build/chrome-mv3-prod](/Users/long.vo/workspaces/personal/side-projects/llm-extensions/build/chrome-mv3-prod)

## 7. Test nhanh

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

## 8. Can thay gi trong diagnostics

### Local-only

- `Provider: ollama`
- `Model`
- `Budget policy`
- `Retrieval time`
- `Generation time`

### Hybrid

- `Provider mode: hybrid`
- `Provider: openai`
- `Hybrid prefilter: applied` hoac `fallback`
- `Prefilter model`
- `Prefilter chunks`
- `Prefilter time`

## 9. Loi thuong gap

### Ollama khong duoc goi

Check:
- Ollama dang chay
- `OLLAMA_BASE_URL` dung
- model da `pull`

### Embeddings fail

Dung:

```env
EMBEDDING_PROVIDER=local
```

### Hybrid khong co prefilter

Check:
- `LLM_PROVIDER=hybrid`
- `HYBRID_PREFILTER_ENABLED=true`
- `OLLAMA_MODEL_SUMMARY` khong rong

## 10. Lenh huu ich

```bash
node scripts/check-env.mjs backend/.env
pnpm benchmark:phase3d
pnpm benchmark:phase3d -- --base-url http://127.0.0.1:8000 --runs 3
```
