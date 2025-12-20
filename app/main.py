"""FastAPI application entry point for Veridict."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    settings = get_settings()
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"📍 Environment: {settings.app_env}")
    
    # TODO: Initialize database connection pool
    # TODO: Validate API keys
    
    yield
    
    # Shutdown
    print("👋 Shutting down Veridict...")
    # TODO: Close database connections


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Zero-Error Legal Due Diligence Engine using the MAKER Framework",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


# Import and include routers
from app.api.routes import documents, analysis
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
