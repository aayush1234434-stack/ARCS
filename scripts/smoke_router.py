#!/usr/bin/env python3
"""Smoke-test router inference for torch or ONNX backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.router.classifier import clear_cache, route


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the domain router (torch or ONNX backend).",
    )
    parser.add_argument(
        "--query",
        default="test query",
        help='Query text to classify (default: "test query")',
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "onnx"),
        default=None,
        help=(
            "Inference backend (default: ARCS_ROUTER_BACKEND env, else torch)"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=config.ROUTER_MODEL_DIR,
        help=f"Router checkpoint directory (default: {config.ROUTER_MODEL_DIR})",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Print JSON only (no progress logs)",
    )
    args = parser.parse_args()

    if args.quiet:
        from arcs import progress

        progress.set_verbose(False)

    clear_cache()
    result = route(args.query, model_dir=str(args.model_dir), backend=args.backend)

    required = {"domain", "confidence", "all_scores", "backend", "model"}
    missing = required - set(result)
    if missing:
        print(f"Error: route result missing keys: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(result["all_scores"], dict) or not result["all_scores"]:
        print("Error: all_scores must be a non-empty dict", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))
    print(
        f"OK backend={result['backend']} domain={result['domain']} "
        f"confidence={result['confidence']:.4f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
