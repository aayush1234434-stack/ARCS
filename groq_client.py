"""Lazy Groq client — defers the slow SDK import until first API call."""

from __future__ import annotations

_client = None


def get_client():
    global _client
    if _client is not None:
        return _client

    import progress

    progress.log("  Loading Groq SDK (first use can take 20s)...")

    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    _client = Groq()
    progress.log("  Groq client ready.")
    return _client
