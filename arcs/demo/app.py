"""
FastAPI demo server for ARCS presentations.

Wraps the existing pipeline, feedback, and logger — no changes to orchestration logic.
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from arcs import config, progress
from arcs.main import run_pipeline
from arcs.post import attribution, feedback, log_entry

STATIC_DIR = Path(__file__).resolve().parent / "static"

load_dotenv(config.PROJECT_ROOT / ".env")

logger = logging.getLogger("arcs.demo")

app = FastAPI(
    title="ARCS Demo",
    description="Adaptive Routing & Correction System — presentation UI",
    version="0.1.0",
)

# Suppress stderr progress spam during API calls (JSON/UI stays clean).
progress.set_verbose(False)


def _check_api_keys() -> None:
    missing: list[str] = []
    if not os.getenv("GROQ_API_KEY", "").strip():
        missing.append("GROQ_API_KEY")
    if not os.getenv("NVIDIA_API_KEY", "").strip():
        missing.append("NVIDIA_API_KEY")
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Missing API keys in .env: {', '.join(missing)}",
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
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
async def api_query(body: QueryRequest) -> QueryResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    _check_api_keys()

    try:
        state = await run_in_threadpool(run_pipeline, query)
    except Exception as exc:
        logger.error("pipeline failed for query=%r\n%s", query[:80], traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc

    record = feedback.apply(state, None)
    try:
        entry = log_entry(record)
    except (TypeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"logging error: {exc}") from exc

    return _summarize_entry(entry)


@app.post("/api/feedback", response_model=FeedbackResponse)
def api_feedback(body: FeedbackRequest) -> FeedbackResponse:
    try:
        record = attribution.load_record(body.query_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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
    except (TypeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"logging error: {exc}") from exc

    attr = enriched.get("attribution")
    if body.signal == "NEGATIVE" and isinstance(attr, dict):
        component = attr.get("component", "UNKNOWN")
        if labeled_domain:
            message = (
                f"Thanks — blame assigned to {component}. "
                f"Labeled correct domain: {labeled_domain}."
            )
        else:
            message = f"Thanks — blame assigned to {component}."
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
