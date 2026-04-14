# Web Context Assistant

Browser extension + FastAPI backend for building a shared context basket from Jira, Confluence, and other web pages, then asking the assistant to summarize, critique, and visualize that context.

## What It Does

- Pin pages into a reusable context basket
- Answer questions over pinned documents with a RAG pipeline
- Suggest related docs from links on the current page
- Generate Mermaid diagrams with export and fullscreen viewing
- Review documents for gaps, risks, and missing requirements

## Repo Layout

- `sidepanel.tsx`, `background.ts`, `contents/`: extension app
- `components/`, `lib/`, `assets/`: shared frontend code
- `backend/`: FastAPI API, retrieval pipeline, LLM integration
- `docs/`: plans and feature notes

## Quick Start

### 1. Install frontend dependencies

```bash
pnpm install
```

### 2. Configure extension env

```bash
cp .env.example .env.local
```

Default:

```env
PLASMO_PUBLIC_API_BASE_URL=http://localhost:8000
```

### 3. Configure backend env

```bash
cd backend
cp .env.example .env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set at least one provider key in `backend/.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key-here
```

or

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key-here
```

For the internal gateway flow, use:

```env
LLM_PROVIDER=openai
AWS_GATEWAY_URL=https://genai-gateway.flava-cloud.com/
PAT_TOKEN=your-internal-pat
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
REQUEST_TIMEOUT_SECONDS=120
```

### 4. Run backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 5. Run extension

```bash
pnpm dev
```

Load the generated unpacked extension into Chrome from the Plasmo dev output.

## Docker

The backend can run in Docker. The extension itself still runs in Chrome, but this repo also includes a Docker-based builder that outputs the packaged extension files.

### Docker onboarding

Shortcut flow:

```bash
make check-env
make docker-up
make docker-smoke
make docker-build-extension
```

Manual flow:

1. Copy compose variables:

```bash
cp .env.compose.example .env.compose
```

2. Copy backend secrets/config:

```bash
cp backend/.env.example backend/.env
```

3. Update `backend/.env` with your provider config
4. Start the backend:

```bash
docker compose --env-file .env.compose up --build -d backend
```

5. Optional smoke check:

```bash
docker compose --env-file .env.compose --profile check up backend-smoke
```

6. Build the extension artifact:

```bash
docker compose --env-file .env.compose --profile build run --rm extension-builder
```

7. Load the unpacked extension from `docker-dist/chrome-mv3-prod` into Chrome

### Makefile shortcuts

```bash
make docker-setup
make check-env
make docker-up
make docker-smoke
make docker-build-extension
make docker-logs
make docker-down
```

What they do:
- `make docker-setup`: create `.env.compose` and `backend/.env` from examples if missing
- `make check-env`: validate `backend/.env` before Docker startup
- `make docker-up`: build and start the backend
- `make docker-smoke`: call the backend health endpoint through the compose network
- `make docker-build-extension`: build the Chrome extension artifact into `docker-dist/`
- `make docker-logs`: tail backend logs
- `make docker-down`: stop the compose stack

### Run backend with Docker Compose

1. Configure [backend/.env](/Users/long.vo/workspaces/personal/side-projects/llm-extensions/backend/.env)
2. Start the API:

```bash
docker compose --env-file .env.compose up --build -d backend
```

The backend will be available at `http://localhost:8000` by default, or the port set in `.env.compose`.

The compose stack now includes:
- a backend `healthcheck`
- a `backend-smoke` service that waits for `service_healthy`
- `extension-builder` depending on the backend health state for a cleaner internal workflow

### Build the extension with Docker Compose

```bash
docker compose --env-file .env.compose --profile build run --rm extension-builder
```

This writes the unpacked production build and packaged zip into `docker-dist/` by default, or the directory set in `.env.compose`.

### Recommended Docker flow for internal users

1. Run `cp .env.compose.example .env.compose`
2. Run `cp backend/.env.example backend/.env`
3. Fill in `backend/.env`
4. Run `docker compose --env-file .env.compose up --build -d backend`
5. Run `docker compose --env-file .env.compose --profile check up backend-smoke`
6. Run `docker compose --env-file .env.compose --profile build run --rm extension-builder`
7. Load the extracted build from `docker-dist/chrome-mv3-prod` in Chrome
8. Point the extension to the backend URL with `PLASMO_PUBLIC_API_BASE_URL`

## Packaging For A Few Users

### Option A: Load unpacked

Best for internal testing.

1. Run `pnpm build`
2. Open Chrome extensions page
3. Enable Developer Mode
4. Load the generated `build/chrome-mv3-prod`

### Option B: Share a packaged zip

```bash
pnpm package
```

This creates `build/web-context-assistant-<version>-chrome-mv3-prod.zip`.

Users should extract the zip first, then load the extracted folder in Chrome as an unpacked extension.

## Chrome Install Guide

### For developers running locally

1. Run `pnpm build`
2. Open Chrome and go to `chrome://extensions`
3. Turn on `Developer mode`
4. Click `Load unpacked`
5. Select [build/chrome-mv3-prod](/Users/long.vo/workspaces/personal/side-projects/llm-extensions/build/chrome-mv3-prod)
6. After the extension appears, click the puzzle icon in Chrome and pin `Web Context Assistant`
7. Open the side panel from the extension icon and verify it can reach the backend

### For internal users receiving a zip

1. Download `web-context-assistant-<version>-chrome-mv3-prod.zip`
2. Extract the zip to a normal folder on disk
3. Open Chrome and go to `chrome://extensions`
4. Turn on `Developer mode`
5. Click `Load unpacked`
6. Select the extracted folder
7. Pin the extension from the Chrome toolbar and open the side panel

### Updating to a newer build

1. Replace the old extracted folder with the new one, or extract the new zip to a new folder
2. Open `chrome://extensions`
3. Find `Web Context Assistant`
4. Click `Reload`

Users still need access to a running backend URL that matches `PLASMO_PUBLIC_API_BASE_URL`.

## Useful Commands

```bash
pnpm dev
pnpm build
pnpm package
pnpm package:plasmo
pnpm typecheck
```

Backend:

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Release Checklist

1. Confirm `.env.local` points to the intended backend URL.
2. Confirm `backend/.env` has valid API credentials.
3. Run `pnpm typecheck`.
4. Smoke test:
   - summarize
   - diagram generation
   - critique mode
   - Mind Reader suggestions
5. Run `pnpm package` and share the generated zip or the unpacked production build.

## Notes

- The extension defaults to `http://localhost:8000` if no frontend env override is provided.
- The backend supports both standard `OPENAI_API_KEY` and internal `PAT_TOKEN + AWS_GATEWAY_URL` for OpenAI-compatible gateways.
- The Docker backend image installs optional local embedding and reranker dependencies too, so `EMBEDDING_PROVIDER=local` works there as well.
- `Clear Cache` in the UI also clears saved diagram cache.
- Generated folders like `build/`, `.plasmo/`, and Python/TS caches should not be committed.
