"""
FastAPI demo server for ARCS presentations.

Wraps the existing pipeline, feedback, and logger — no changes to orchestration logic.

Production hardening (demo UI only):
  - In-memory per-IP rate limits on /api/query and /api/feedback
  - Pipeline wall-clock timeout (avoids hung Groq calls blocking forever)
  - Sanitized 429 / 503 / 500 JSON errors (no stack traces to clients)
  - Optional ARCS_DEMO_PUBLIC=1 disclaimer banner via /health
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from arcs import config, progress
from arcs.main import PipelineError, run_pipeline
from arcs.demo.mock import run_pipeline as run_mock_pipeline
from arcs.post import attribution, feedback, log_entry

STATIC_DIR = Path(__file__).resolve().parent / "static"

load_dotenv(config.PROJECT_ROOT / ".env")

logger = logging.getLogger("arcs.demo")

# ── Demo ops knobs (env) ──────────────────────────────────────────────────────

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


# Requests per window for /api/query (0 = disabled).
DEMO_QUERY_RATE_LIMIT = _env_int("ARCS_DEMO_RATE_LIMIT", 8)
DEMO_QUERY_RATE_WINDOW = _env_float("ARCS_DEMO_RATE_WINDOW", 60.0)
# Feedback is cheaper; allow more (0 = disabled).
DEMO_FEEDBACK_RATE_LIMIT = _env_int("ARCS_DEMO_FEEDBACK_RATE_LIMIT", 30)
DEMO_FEEDBACK_RATE_WINDOW = _env_float("ARCS_DEMO_FEEDBACK_RATE_WINDOW", 60.0)
# Wall-clock timeout for a single pipeline call (seconds).
DEMO_PIPELINE_TIMEOUT = _env_float("ARCS_DEMO_PIPELINE_TIMEOUT", 180.0)
DEMO_PUBLIC = _env_bool("ARCS_DEMO_PUBLIC", False)
DEMO_OFFLINE = _env_bool("ARCS_DEMO_OFFLINE", False)
DEMO_TRUST_PROXY_HEADERS = _env_bool("ARCS_TRUST_PROXY_HEADERS", False)

PUBLIC_DISCLAIMER = (
    "Public demo — educational use only. Answers may be wrong. "
    "Not medical, legal, or professional advice. Rate limits apply."
)

app = FastAPI(
    title="ARCS Demo",
    description="Adaptive Routing & Correction System — presentation UI",
    version="0.1.0",
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """Apply a restrictive browser policy to every demo response."""
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    )
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/") or request.url.path == "/health":
        response.headers["Cache-Control"] = "no-store"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Suppress stderr progress spam during API calls (JSON/UI stays clean).
progress.set_verbose(False)


class InMemoryRateLimiter:
    """Sliding-window counter keyed by client IP (process-local; no Redis)."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_requests > 0 and self.window_seconds > 0

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        if not self.enabled:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            cutoff = now - self.window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(0.0, self.window_seconds - (now - bucket[0]))
                return False, retry_after
            bucket.append(now)
            return True, 0.0


_query_limiter = InMemoryRateLimiter(DEMO_QUERY_RATE_LIMIT, DEMO_QUERY_RATE_WINDOW)
_feedback_limiter = InMemoryRateLimiter(
    DEMO_FEEDBACK_RATE_LIMIT, DEMO_FEEDBACK_RATE_WINDOW
)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and DEMO_TRUST_PROXY_HEADERS:
        # First hop is the original client when behind a reverse proxy.
        return forwarded.split(",")[0].strip() or "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(limiter: InMemoryRateLimiter, request: Request) -> None:
    allowed, retry_after = limiter.check(_client_ip(request))
    if allowed:
        return
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded. Please wait and try again.",
        headers={"Retry-After": str(max(1, int(retry_after) + 1))},
    )


def _check_api_keys() -> None:
    if DEMO_OFFLINE:
        return
    missing: list[str] = []
    if not os.getenv("GROQ_API_KEY", "").strip():
        missing.append("GROQ_API_KEY")
    if not os.getenv("NVIDIA_API_KEY", "").strip():
        missing.append("NVIDIA_API_KEY")
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: required API keys are not configured.",
        )


class HealthResponse(BaseModel):
    status: str
    groq_configured: bool
    nvidia_configured: bool
    router_backend: str
    public_demo: bool = False
    disclaimer: str | None = None
    demo_mode: str = "live"
    secure_sandbox_required: bool = True


def _health_payload() -> HealthResponse:
    return HealthResponse(
        status="ok",
        groq_configured=bool(os.getenv("GROQ_API_KEY", "").strip()),
        nvidia_configured=bool(os.getenv("NVIDIA_API_KEY", "").strip()),
        router_backend=config.ROUTER_BACKEND,
        public_demo=DEMO_PUBLIC,
        disclaimer=PUBLIC_DISCLAIMER if DEMO_PUBLIC else None,
        demo_mode="offline" if DEMO_OFFLINE else "live",
        secure_sandbox_required=True,
    )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    domain: str | None
    confidence: float | None
    pipeline_id: str | None
    verifier: str | None
    verdict: str | None
    score: float | None
    status: str
    timing_ms: int | None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    demo_mode: str = "live"


VALID_FEEDBACK_DOMAINS = frozenset({"CODING", "MEDICAL", "LEGAL", "GENERAL"})


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    signal: Literal["POSITIVE", "NEGATIVE"]
    correct_domain: Literal["CODING", "MEDICAL", "LEGAL", "GENERAL"] | None = None


class FeedbackResponse(BaseModel):
    query_id: str
    signal: str
    attribution: dict[str, Any] | None
    correct_domain: str | None = None
    message: str


def _build_trace(entry: dict[str, Any]) -> list[dict[str, Any]]:
    route = entry.get("route") or {}
    pipeline = entry.get("pipeline") or {}
    specification = entry.get("specification") or {}
    verification = entry.get("verification") or {}
    tooling = entry.get("tooling") or {}
    timing = entry.get("timing") or {}
    confidence = route.get("confidence")
    confidence_text = (
        f"{100 * float(confidence):.0f}% confidence"
        if isinstance(confidence, (int, float))
        else "confidence unavailable"
    )
    required = specification.get("required_elements") or []
    return [
        {
            "stage": "route",
            "label": "Route",
            "status": "complete",
            "detail": f"{route.get('domain', 'UNKNOWN')} · {confidence_text}",
            "timing_ms": timing.get("route_ms"),
        },
        {
            "stage": "specification",
            "label": "Specify",
            "status": "complete" if specification else "skipped",
            "detail": f"{len(required)} required element(s) · {specification.get('model', 'unavailable')}",
            "timing_ms": timing.get("specification_ms"),
        },
        {
            "stage": "generation",
            "label": "Generate",
            "status": "complete",
            "detail": str(pipeline.get("generator_model") or "model unavailable"),
            "timing_ms": timing.get("specialist_ms"),
        },
        {
            "stage": "verification",
            "label": "Verify",
            "status": str(verification.get("verdict") or "unknown").lower(),
            "detail": (
                f"{pipeline.get('verifier', 'unknown')} · score "
                f"{verification.get('score', 'n/a')} · {tooling.get('rounds_used', 1)} round(s)"
            ),
            "timing_ms": timing.get("verification_ms"),
        },
    ]


def _summarize_entry(entry: dict[str, Any]) -> QueryResponse:
    route = entry.get("route") or {}
    pipeline = entry.get("pipeline") or {}
    verification = entry.get("verification") or {}
    timing = entry.get("timing") or {}
    total_ms = timing.get("total_ms") if isinstance(timing, dict) else None

    confidence = route.get("confidence")
    try:
        confidence_f = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_f = None

    score = verification.get("score")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None

    return QueryResponse(
        query_id=str(entry.get("query_id", "")),
        query=str(entry.get("query", "")),
        answer=str(entry.get("response") or (entry.get("specialist") or {}).get("answer", "")),
        domain=route.get("domain"),
        confidence=confidence_f,
        pipeline_id=pipeline.get("pipeline_id"),
        verifier=pipeline.get("verifier"),
        verdict=verification.get("verdict"),
        score=score_f,
        status=str(entry.get("status", "UNKNOWN")),
        timing_ms=int(total_ms) if isinstance(total_ms, (int, float)) else None,
        trace=_build_trace(entry),
        usage=entry.get("usage") if isinstance(entry.get("usage"), dict) else {},
        demo_mode=str(entry.get("demo_mode") or ("offline" if DEMO_OFFLINE else "live")),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak stack traces or internal exception text to clients."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=dict(exc.headers) if exc.headers else None,
        )
    logger.error(
        "unhandled error path=%s\n%s",
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return _health_payload()


@app.post("/api/query", response_model=QueryResponse)
async def api_query(body: QueryRequest, request: Request) -> QueryResponse:
    _enforce_rate_limit(_query_limiter, request)

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    _check_api_keys()

    try:
        state = await asyncio.wait_for(
            run_in_threadpool(run_mock_pipeline if DEMO_OFFLINE else run_pipeline, query),
            timeout=DEMO_PIPELINE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "pipeline timeout after %.0fs query=%r",
            DEMO_PIPELINE_TIMEOUT,
            query[:80],
        )
        raise HTTPException(
            status_code=503,
            detail="Pipeline timed out. Please try again shortly.",
        ) from None
    except PipelineError as exc:
        logger.error(
            "pipeline failed query=%r error_class=%s\n%s",
            query[:80],
            (exc.state or {}).get("error_class"),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=503,
            detail="Pipeline unavailable. Please try again shortly.",
        ) from None
    except HTTPException:
        raise
    except Exception:
        logger.error("pipeline failed query=%r\n%s", query[:80], traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Pipeline failed. Please try again.",
        ) from None

    record = feedback.apply(state, None)
    try:
        entry = log_entry(record)
    except (TypeError, OSError):
        logger.error("logging error\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Could not record result. Please try again.",
        ) from None

    return _summarize_entry(entry)


@app.post("/api/feedback", response_model=FeedbackResponse)
def api_feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    _enforce_rate_limit(_feedback_limiter, request)

    try:
        record = attribution.load_record(body.query_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown query_id.") from None
    except LookupError:
        raise HTTPException(status_code=404, detail="Unknown query_id.") from None

    if body.correct_domain is not None and body.signal != "NEGATIVE":
        raise HTTPException(
            status_code=400,
            detail="correct_domain is only allowed with NEGATIVE feedback",
        )

    signal = feedback.collect(explicit=body.signal)
    enriched = feedback.apply(record, signal)
    enriched["query_id"] = body.query_id

    # Optional router-retrain label. Attribution rules are unchanged.
    labeled_domain: str | None = None
    if body.signal == "NEGATIVE" and body.correct_domain is not None:
        labeled_domain = body.correct_domain.strip().upper()
        if labeled_domain not in VALID_FEEDBACK_DOMAINS:
            raise HTTPException(
                status_code=400,
                detail=f"correct_domain must be one of {sorted(VALID_FEEDBACK_DOMAINS)}",
            )
        enriched["correct_domain"] = labeled_domain
        enriched["expected_domain"] = labeled_domain

    try:
        log_entry(enriched)
    except (TypeError, OSError):
        logger.error("feedback logging error\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Could not record feedback. Please try again.",
        ) from None

    attr = enriched.get("attribution")
    if body.signal == "NEGATIVE" and isinstance(attr, dict):
        component = attr.get("component", "UNKNOWN")
        if labeled_domain:
            message = (
                f"Thanks — blame assigned to {component}. "
                f"Labeled correct domain: {labeled_domain}."
            )
        else:
            message = (
                f"Thanks — blame assigned to {component}. "
                "Pick a domain next time if routing was wrong (feeds router retrain)."
            )
    elif body.signal == "POSITIVE":
        message = "Thanks for the positive feedback!"
    else:
        message = "Feedback recorded."

    return FeedbackResponse(
        query_id=body.query_id,
        signal=body.signal,
        attribution=attr if isinstance(attr, dict) else None,
        correct_domain=labeled_domain,
        message=message,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
