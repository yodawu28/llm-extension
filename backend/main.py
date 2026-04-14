import logging
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from routers import summarize

settings = get_settings()


def _configure_logging() -> None:
    logging.basicConfig(
        level=settings.resolved_log_level(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )


_configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="AI-powered assistant API for summarizing web pages",
    version="0.1.0",
    debug=settings.debug,
)

# CORS configuration for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(summarize.router)


@app.on_event("startup")
async def log_startup_configuration():
    logger.info("Backend startup configuration: %s", settings.diagnostics_summary())

    if settings.llm_provider == "openai" and not settings.resolved_openai_api_key():
        logger.warning("No OpenAI API key or PAT token configured for the active OpenAI-compatible provider.")

    if settings.llm_provider == "openai" and not settings.resolved_openai_base_url() and settings.pat_token:
        logger.warning("PAT token is configured without an OpenAI-compatible base URL or gateway URL.")


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.exception(
            "HTTP request failed method=%s path=%s duration_ms=%s",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        "HTTP request completed method=%s path=%s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a client-friendly 400 instead of a raw FastAPI 422."""
    error_messages = []

    for error in exc.errors():
        location = " -> ".join(str(part) for part in error.get("loc", []) if part != "body")
        message = error.get("msg", "Invalid value")
        error_messages.append(f"{location}: {message}" if location else message)

    detail = "; ".join(error_messages) or "Invalid request payload"
    logger.warning(
        "Request validation failed method=%s path=%s detail=%s",
        request.method,
        request.url.path,
        detail,
    )
    return JSONResponse(status_code=400, content={"detail": detail})


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Web Context Assistant API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.debug,
    )
