"""FastAPI server for Clockify RAG system.

Provides REST API endpoints:
- GET /health: Health check
- GET /v1/config: Current configuration
- POST /v1/query: Submit a question
- POST /v1/ingest: Trigger index build
- GET /v1/metrics: System metrics (JSON/Prometheus/CSV via format param)
- GET /metrics: Standard Prometheus scraping endpoint
"""

import asyncio
import json
import logging
import os
import platform
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import config
from .answer import answer_once
from .caching import get_rate_limiter as _get_rate_limiter
from .runtime import ensure_index_ready
from .correlation import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    validate_correlation_id,
)
from .exceptions import ValidationError
from .indexing import build, index_is_fresh
from .metrics import MetricNames, get_metrics
from .utils import ALLOWED_CORPUS_FILENAME, check_ollama_connectivity, resolve_corpus_path

# Re-export for tests that monkeypatch api.get_rate_limiter
get_rate_limiter = _get_rate_limiter

logger = logging.getLogger(__name__)


def _threadpool_workers() -> int:
    """Compute a threadpool size that can handle small concurrent bursts."""

    cpu_count = os.cpu_count() or 1
    return max(4, min(32, cpu_count * 4))


# ============================================================================
# Pydantic Models
# ============================================================================


class QueryRequest(BaseModel):
    """Request body for /v1/query endpoint."""

    question: str = Field(..., min_length=1, max_length=config.MAX_QUERY_LENGTH, description="Question to answer")
    top_k: Optional[int] = Field(
        config.DEFAULT_TOP_K, ge=1, le=config.MAX_TOP_K, description="Number of chunks to retrieve"
    )
    pack_top: Optional[int] = Field(config.DEFAULT_PACK_TOP, ge=1, le=50, description="Number of chunks in context")
    threshold: Optional[float] = Field(config.DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Minimum similarity")
    debug: Optional[bool] = Field(False, description="Include debug information")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate and sanitize question input.

        Prevents XSS, injection attacks, and other malicious input.
        """
        # Strip excessive whitespace
        v = " ".join(v.split())

        if not v:
            raise ValueError("Question cannot be empty after whitespace removal")

        # Check for suspicious patterns (basic XSS prevention)
        suspicious_patterns = [
            "<script",
            "javascript:",
            "onerror=",
            "onload=",
            "<iframe",
            "eval(",
            "expression(",
        ]

        v_lower = v.lower()
        for pattern in suspicious_patterns:
            if pattern in v_lower:
                raise ValueError("Invalid content detected in question")

        # Ensure only printable characters (allow unicode for i18n)
        if not all(c.isprintable() or c.isspace() for c in v):
            raise ValueError("Question contains non-printable characters")

        return v


class Citation(BaseModel):
    """Rich citation details for UI rendering."""

    id: str
    title: Optional[str] = None
    url: Optional[str] = None
    section: Optional[str] = None
    breadcrumb: Optional[str] = None
    article_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response body for /v1/query endpoint."""

    question: str
    answer: str
    confidence: Optional[float] = None
    sources: list[str] = Field(default_factory=list, description="Chunk IDs used")
    timestamp: datetime
    processing_time_ms: float
    refused: bool = Field(False, description="True when the answer is a refusal/fallback")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (confidence routing, errors)")
    routing: Optional[Dict[str, Any]] = Field(None, description="Routing recommendation (if available)")
    timing: Optional[Dict[str, Any]] = Field(None, description="Latency breakdown in milliseconds")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID for tracing")
    citations: Optional[list[Citation]] = Field(None, description="Rich citation details")


class HealthResponse(BaseModel):
    """Response body for /v1/health endpoint."""

    status: str
    timestamp: datetime
    version: str
    platform: str
    index_ready: bool
    ollama_connected: bool


class ConfigResponse(BaseModel):
    """Response body for /v1/config endpoint."""

    ollama_url: str
    gen_model: str
    emb_model: str
    chunk_size: int
    top_k: int
    pack_top: int
    threshold: float


class IngestRequest(BaseModel):
    """Request body for /v1/ingest endpoint."""

    input_file: Optional[str] = Field(None, description="Input markdown file")
    force: Optional[bool] = Field(False, description="Force rebuild")


class IngestResponse(BaseModel):
    """Response body for /v1/ingest endpoint."""

    status: str
    message: str
    timestamp: datetime
    index_ready: bool


# ============================================================================
# FastAPI Application
# ============================================================================


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    def _clear_index_state(target_app: FastAPI) -> None:
        """Clear index state with thread-safe locking."""
        with target_app.state.lock:
            target_app.state.chunks = None
            target_app.state.vecs_n = None
            target_app.state.bm = None
            target_app.state.hnsw = None
            target_app.state.index_ready = False

    def _set_index_state(target_app: FastAPI, result) -> None:
        """Set index state with thread-safe locking to prevent race conditions."""
        with target_app.state.lock:
            if not result:
                # Must release lock before calling _clear_index_state since it acquires same lock
                pass
            else:
                chunks, vecs_n, bm, hnsw = result
                target_app.state.chunks = chunks
                target_app.state.vecs_n = vecs_n
                target_app.state.bm = bm
                target_app.state.hnsw = hnsw
                target_app.state.index_ready = True

        # Call clear outside lock if needed (RLock is reentrant so this is safe, but clearer)
        if not result:
            _clear_index_state(target_app)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Ensure a sufficiently-sized executor for run_in_executor workloads (queries/ingest)
        executor = ThreadPoolExecutor(max_workers=_threadpool_workers())
        asyncio.get_running_loop().set_default_executor(executor)
        _app.state.executor = executor
        try:
            logger.info("Loading index on startup...")
            try:
                result = ensure_index_ready(retries=2)
                if result:
                    _set_index_state(_app, result)
                    logger.info("Index loaded: %d chunks", len(result[0]))
                else:
                    logger.warning("Index not ready at startup")
            except Exception as exc:
                logger.error("Failed to load index at startup: %s", exc)
                _clear_index_state(_app)
            yield
        finally:
            logger.info("Initiating graceful shutdown...")
            executor.shutdown(wait=True)
            _clear_index_state(_app)
            logger.info("Graceful shutdown complete")

    app = FastAPI(
        title="Clockify RAG API",
        description="Production-ready RAG system with hybrid retrieval",
        version="5.9.1",
        lifespan=lifespan,
    )

    # Initialize thread-safety lock on app.state (prevents race conditions during ingest)
    # Using RLock for reentrant locking support
    app.state.lock = threading.RLock()
    # Serialize ingest builds without blocking query reads on app.state.lock
    app.state.ingest_lock = threading.Lock()

    # Add CORS middleware only when explicitly configured
    if config.ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=[
                config.API_KEY_HEADER or "x-api-key",
                "content-type",
                "accept",
                "x-correlation-id",
                "x-request-id",
            ],
        )

    # ========================================================================
    # Correlation ID Middleware
    # ========================================================================
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        """Add correlation ID to request context for distributed tracing.

        Extracts correlation ID from incoming headers (X-Correlation-ID or X-Request-ID)
        or generates a new one. Validates input to prevent log injection.
        """
        # Extract and validate from headers
        raw_id = request.headers.get("x-correlation-id") or request.headers.get("x-request-id")
        correlation_id = validate_correlation_id(raw_id) or generate_correlation_id()

        # Set in context for logging (propagates to thread pool via ContextVar)
        set_correlation_id(correlation_id)
        # Also store on request.state so exception handlers can access it
        # after the ContextVar is cleared in finally
        request.state.correlation_id = correlation_id

        try:
            response = await call_next(request)
            # Add to response headers for client tracing
            response.headers["x-correlation-id"] = correlation_id
            return response
        finally:
            # Clear context after request completes
            clear_correlation_id()

    _clear_index_state(app)

    def _require_api_key(req: Request) -> None:
        if config.API_AUTH_MODE != "api_key":
            return
        header_name = config.API_KEY_HEADER or "x-api-key"
        api_key = req.headers.get(header_name)
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key")
        if api_key not in config.API_ALLOWED_KEYS:
            raise HTTPException(status_code=403, detail="Invalid API key")

    # ========================================================================
    # Exception Handlers (ensure correlation ID on error responses)
    # ========================================================================
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with correlation ID header."""
        # Read from request.state first (survives middleware finally block),
        # fall back to ContextVar, then generate new if neither available
        correlation_id = (
            getattr(request.state, "correlation_id", None) or get_correlation_id() or generate_correlation_id()
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers={"x-correlation-id": correlation_id},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions with correlation ID header."""
        # Read from request.state first (survives middleware finally block),
        # fall back to ContextVar, then generate new if neither available
        correlation_id = (
            getattr(request.state, "correlation_id", None) or get_correlation_id() or generate_correlation_id()
        )
        logger.exception(f"Unhandled exception [correlation_id={correlation_id}]: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
            headers={"x-correlation-id": correlation_id},
        )

    # ========================================================================
    # Health Check Endpoint
    # ========================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Enhanced health check endpoint with dependency validation.

        Returns system status, index readiness, and Ollama connectivity.
        Status levels:
        - healthy: Index ready and Ollama connected
        - degraded: Index ready but Ollama unavailable (can serve cached queries)
        - unavailable: Index not ready (cannot serve any queries)
        """
        from . import __version__
        from pathlib import Path

        # Check index files exist (belt-and-suspenders with app.state)
        index_files_exist = all(Path(f).exists() for f in ["chunks.jsonl", "vecs_n.npy", "meta.jsonl", "bm25.json"])

        # Read index_ready atomically
        with app.state.lock:
            index_ready = app.state.index_ready and index_files_exist

        # Check Ollama connectivity with short timeout (threadpool to avoid blocking)
        ollama_ok = False
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, partial(check_ollama_connectivity, config.RAG_OLLAMA_URL or "", 2))
            ollama_ok = True
        except Exception as e:
            # Ollama connectivity failure is acceptable for health check
            # (allows graceful degradation), but log for debugging
            logger.debug(f"Ollama health check failed: {e}")

        # Determine overall status
        if not index_ready:
            status = "unavailable"
        elif index_ready and ollama_ok:
            status = "healthy"
        else:
            status = "degraded"

        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            version=__version__,
            platform=f"{platform.system()} {platform.machine()}",
            index_ready=index_ready,
            ollama_connected=ollama_ok,
        )

    @app.get("/v1/health", response_model=HealthResponse)
    async def health_check_v1() -> HealthResponse:
        """Health check endpoint (v1 API)."""
        return await health_check()

    # ========================================================================
    # Configuration Endpoint
    # ========================================================================

    @app.get("/v1/config", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        """Get current configuration."""
        return ConfigResponse(
            ollama_url=config.RAG_OLLAMA_URL or "",
            gen_model=config.RAG_CHAT_MODEL or "",
            emb_model=config.RAG_EMBED_MODEL or "",
            chunk_size=config.CHUNK_CHARS,
            top_k=config.DEFAULT_TOP_K,
            pack_top=config.DEFAULT_PACK_TOP,
            threshold=config.DEFAULT_THRESHOLD,
        )

    # ========================================================================
    # Query Endpoint
    # ========================================================================

    @app.post("/v1/query", response_model=QueryResponse)
    async def submit_query(request: QueryRequest, raw_request: Request) -> QueryResponse:
        """Submit a question and get an answer.

        This endpoint uses the RAG system to retrieve relevant context
        and generate an answer using the LLM.

        Args:
            request: QueryRequest with question and parameters

        Returns:
            QueryResponse with answer, confidence, and sources

        Raises:
            HTTPException: If index not ready or query fails
        """
        _require_api_key(raw_request)

        # Check index readiness and capture state atomically
        with app.state.lock:
            if not app.state.index_ready:
                raise HTTPException(
                    status_code=503, detail="Index not ready. Run /v1/ingest first or wait for startup."
                )

            # Capture state references atomically to prevent mid-query state changes
            chunks = app.state.chunks
            vecs_n = app.state.vecs_n
            bm = app.state.bm
            hnsw = app.state.hnsw

        try:
            start_time = time.time()
            metrics = get_metrics()
            rate_limiter = get_rate_limiter()
            if rate_limiter:
                if rate_limiter.allow_request():
                    metrics.increment_counter(MetricNames.RATE_LIMIT_ALLOWED)
                else:
                    metrics.increment_counter(MetricNames.RATE_LIMIT_BLOCKED)
                    wait_seconds = 0.0
                    if hasattr(rate_limiter, "wait_time"):
                        try:
                            wait_seconds = float(rate_limiter.wait_time())
                        except Exception:
                            wait_seconds = 0.0
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded. Retry after {wait_seconds:.2f} seconds.",
                    )

            loop = asyncio.get_running_loop()
            resolved_top_k = int(request.top_k) if request.top_k is not None else config.DEFAULT_TOP_K
            resolved_pack_top = int(request.pack_top) if request.pack_top is not None else config.DEFAULT_PACK_TOP
            resolved_threshold = float(request.threshold) if request.threshold is not None else config.DEFAULT_THRESHOLD

            answer_future = partial(
                answer_once,
                request.question,
                chunks,
                vecs_n,
                bm,
                top_k=resolved_top_k,
                pack_top=resolved_pack_top,
                threshold=resolved_threshold,
                use_rerank=True,
                hnsw=hnsw,
            )
            executor = getattr(app.state, "executor", None)
            result = await loop.run_in_executor(executor, answer_future)

            elapsed_ms = (time.time() - start_time) * 1000

            metadata = result.get("metadata") or {}
            selected_chunks = result.get("selected_chunks", [])
            chunk_ids = result.get("selected_chunk_ids") or selected_chunks
            sources_used = result.get("sources_used") or metadata.get("sources_used") or []
            citation_details = result.get("citation_details") or metadata.get("citation_details")
            if sources_used:
                sources = [str(identifier) for identifier in sources_used][:5]
            else:
                sources = [str(identifier) for identifier in (chunk_ids or [])][:5]

            return QueryResponse(
                question=request.question,
                answer=result["answer"],
                confidence=result.get("confidence"),
                sources=sources,
                timestamp=datetime.now(),
                processing_time_ms=elapsed_ms,
                refused=result.get("refused", False),
                metadata=metadata or {},
                routing=result.get("routing"),
                timing=result.get("timing"),
                correlation_id=get_correlation_id(),
                citations=citation_details,
            )

        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            # ValidationError messages are safe user-facing messages
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            # Re-raise HTTP exceptions (like 429 rate limit) without modification
            raise
        except Exception as e:
            # Generic exceptions may contain internal details - sanitize
            logger.error(f"Query error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    # ========================================================================
    # Ingest Endpoint
    # ========================================================================

    @app.post("/v1/ingest", response_model=IngestResponse)
    async def trigger_ingest(
        request: IngestRequest, raw_request: Request, background_tasks: BackgroundTasks
    ) -> IngestResponse:
        """Trigger index build/rebuild.

        Starts a background task to build the index from the knowledge base.

        Args:
            request: IngestRequest with input file and options
            background_tasks: FastAPI background tasks

        Returns:
            IngestResponse with status

        Note:
            Build happens asynchronously. Check /health to verify completion.
        """
        _require_api_key(raw_request)
        if request.input_file and os.path.basename(request.input_file) != ALLOWED_CORPUS_FILENAME:
            raise HTTPException(
                status_code=400,
                detail=f"Only {ALLOWED_CORPUS_FILENAME} is supported for ingestion.",
            )
        input_file, exists, candidates = resolve_corpus_path(request.input_file)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Input file not found. Looked for: {', '.join(candidates)}")

        with app.state.lock:
            already_ready = app.state.index_ready

        if not request.force and already_ready and index_is_fresh(input_file):
            return IngestResponse(
                status="skipped",
                message=f"Index already up to date for {input_file}",
                timestamp=datetime.now(),
                index_ready=True,
            )

        def do_ingest():
            """Background task to build index."""
            started_at = time.time()
            with app.state.ingest_lock:
                with app.state.lock:
                    prior_ready = app.state.index_ready
                    prior_state = (app.state.chunks, app.state.vecs_n, app.state.bm, app.state.hnsw)
                try:
                    logger.info(f"Starting ingest from {input_file}")
                    build(input_file, retries=2)
                    result = ensure_index_ready(retries=2)
                    _set_index_state(app, result)
                    duration_ms = (time.time() - started_at) * 1000
                    logger.info(f"Ingest completed successfully in {duration_ms:.1f} ms")
                except Exception as e:
                    logger.error(f"Ingest failed: {e}", exc_info=True)
                    if prior_ready:
                        logger.warning("Restoring previous index state after ingest failure")
                        _set_index_state(app, prior_state)
                    else:
                        _clear_index_state(app)

        background_tasks.add_task(do_ingest)

        return IngestResponse(
            status="processing",
            message=f"Index build started in background from {input_file}",
            timestamp=datetime.now(),
            index_ready=app.state.index_ready,
        )

    # ========================================================================
    # Metrics Endpoint
    # ========================================================================

    @app.get("/v1/metrics")
    async def get_metrics_endpoint(format: str = "json") -> Response:
        """Expose in-process metrics in JSON, Prometheus, or CSV format."""
        collector = get_metrics()
        fmt = (format or "json").lower()

        if fmt == "prometheus":
            payload = collector.export_prometheus()
            return Response(payload, media_type="text/plain; version=0.0.4")
        if fmt == "csv":
            payload = collector.export_csv()
            return Response(payload, media_type="text/csv")

        # Default JSON structure
        snapshot_json = collector.export_json(include_histograms=True)
        payload = json.loads(snapshot_json)

        # Read state atomically
        with app.state.lock:
            payload["index_ready"] = app.state.index_ready
            payload["chunks_loaded"] = len(app.state.chunks) if app.state.chunks else 0

        return JSONResponse(payload)

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        """Standard Prometheus metrics endpoint.

        This is the conventional endpoint path for Prometheus scraping.
        Returns metrics in Prometheus text format.

        Example scrape config:
            scrape_configs:
              - job_name: 'clockify-rag'
                static_configs:
                  - targets: ['localhost:8000']
        """
        collector = get_metrics()
        payload = collector.export_prometheus()
        return Response(payload, media_type="text/plain; version=0.0.4; charset=utf-8")

    return app


# ============================================================================
# Standalone Server
# ============================================================================


app = create_app()


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 4,
    log_level: str = "info",
) -> None:
    """Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        log_level: Logging level
    """
    import uvicorn

    uvicorn.run(
        "clockify_rag.api:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
