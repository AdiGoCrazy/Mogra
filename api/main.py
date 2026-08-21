import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from api.routes import router
from logger.unified_logger import configure_logging, get_logger, Subsystem

configure_logging(is_tui=False)
logger = get_logger(Subsystem.API_MAIN)

app = FastAPI(
    title="Mogra Movie Recommender Agent API 🎬🌐",
    description=(
        "Production REST API for the Mogra Movie Recommender Agent.\n\n"
        "Features:\n"
        "- **GraphRAG + 16-Channel Multi-Vector Retrieval**\n"
        "- **Local Qwen 2.5 Intent Parsing & Synthesis**\n"
        "- **Interactive Swagger UI OpenAPI Documentation**\n"
        "- **RLHF Feedback Dataset Collection**"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS for third-party web & mobile UI applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(router)

@app.get("/", include_in_schema=False)
def root():
    """Redirect root path to interactive Swagger UI documentation page."""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Health"], summary="API Healthcheck")
def healthcheck():
    """Healthcheck endpoint returning server operating status."""
    return {"status": "HEALTHY", "service": "Mogra Movie Recommender Agent API"}
