from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from routers import summarize

settings = get_settings()

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


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a client-friendly 400 instead of a raw FastAPI 422."""
    error_messages = []

    for error in exc.errors():
        location = " -> ".join(str(part) for part in error.get("loc", []) if part != "body")
        message = error.get("msg", "Invalid value")
        error_messages.append(f"{location}: {message}" if location else message)

    detail = "; ".join(error_messages) or "Invalid request payload"
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
