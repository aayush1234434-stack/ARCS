"""
Lightweight stderr progress logging for CLI runs.

All status output goes to stderr so stdout stays clean for JSON results.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager

_start = time.monotonic()
_verbose = True


def set_verbose(enabled: bool) -> None:
    global _verbose
    _verbose = enabled


def log(message: str) -> None:
    if not _verbose:
        return
    elapsed = time.monotonic() - _start
    print(f"[{elapsed:6.1f}s] {message}", file=sys.stderr, flush=True)


@contextmanager
def step(name: str):
    log(f"→ {name}...")
    started = time.monotonic()
    try:
        yield
    except Exception:
        log(f"✗ {name} failed after {time.monotonic() - started:.1f}s")
        raise
    else:
        log(f"✓ {name} ({time.monotonic() - started:.1f}s)")
