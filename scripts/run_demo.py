"""Start the ARCS demo web UI (FastAPI + uvicorn)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _default_port() -> int:
    """Respect PLATFORM PORT (Railway/Render/Fly) when set."""
    raw = os.getenv("PORT", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 8000


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ARCS demo web UI.")
    parser.add_argument(
        "--host",
        default=os.getenv("ARCS_DEMO_HOST", "127.0.0.1"),
        help="Bind host (default: 127.0.0.1; Docker/cloud: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_default_port(),
        help="Bind port (default: PORT env or 8000)",
    )
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import uvicorn

    uvicorn.run(
        "arcs.demo.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
