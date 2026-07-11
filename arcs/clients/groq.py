"""Lazy Groq client — defers the slow SDK import until first API call."""

from __future__ import annotations

_client = None


def get_client():
    global _client
    if _client is not None:
        return _client

    from arcs import progress
    from arcs.config import PROJECT_ROOT

    progress.log("  Loading Groq SDK (first use can take 20s)...")

    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(PROJECT_ROOT / ".env")
    # max_retries lets the SDK ride out transient 429 (TPM) and connection
    # errors: it honors the Retry-After header, so a rate-limited call waits the
    # window out instead of surfacing as a hard pipeline error.
    _client = Groq(max_retries=5, timeout=90.0)
    progress.log("  Groq client ready.")
    return _client
