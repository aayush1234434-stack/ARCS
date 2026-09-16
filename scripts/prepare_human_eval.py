#!/usr/bin/env python3
"""Create a randomized, blinded human-evaluation packet from experiment runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arcs import config
from arcs.eval.experiments import load_experiment, sha256_file


def _rows(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    meta = experiment.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("rows"), list):
        return [row for row in meta["rows"] if isinstance(row, dict)]
    for key in ("rows", "results", "items"):
        value = experiment.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _answer(row: dict[str, Any]) -> str:
    specialist = row.get("specialist")
    if isinstance(specialist, dict) and specialist.get("answer"):
        return str(specialist["answer"])
    return str(row.get("answer") or row.get("response") or "")


def _blind_id(seed: int, system: str, row_id: str) -> str:
    value = f"{seed}:{system}:{row_id}".encode()
    return hashlib.sha256(value).hexdigest()[:16]


def prepare(
    systems: list[tuple[str, Path]],
    output_dir: Path,
    *,
    seed: int,
) -> Path:
    """Write reviewer packet, private answer key, and provenance manifest."""
    items: list[dict[str, str]] = []
    answer_key: list[dict[str, str]] = []
    sources: list[dict[str, str]] = []

    for label, path in systems:
        source_json = path / "experiment.json" if path.is_dir() else path
        experiment = load_experiment(path)
        run_rows = _rows(experiment)
        if not run_rows:
            raise ValueError(f"experiment has no row-level evidence: {path}")
        sources.append(
            {"system": label, "path": str(path), "sha256": sha256_file(source_json)}
        )
        for index, row in enumerate(run_rows, start=1):
            row_id = str(row.get("id") or row.get("query_id") or f"row-{index}")
            answer = _answer(row).strip()
            if not answer:
                continue
            item_id = _blind_id(seed, label, row_id)
            items.append(
                {
                    "item_id": item_id,
                    "query": str(row.get("query") or ""),
                    "expected_domain": str(row.get("expected_domain") or ""),
                    "answer": answer,
                    "reviewer": "",
                    "correctness_1_5": "",
                    "completeness_1_5": "",
                    "safety_1_5": "",
                    "citation_quality_1_5": "",
                    "harmful_yes_no": "",
                    "notes": "",
                }
            )
            answer_key.append({"item_id": item_id, "system": label, "source_row_id": row_id})

    rng = random.Random(seed)
    rng.shuffle(items)
    output_dir.mkdir(parents=True, exist_ok=False)
    packet_path = output_dir / "review_packet.csv"
    fieldnames = list(items[0].keys()) if items else []
    if not fieldnames:
        raise ValueError("no answers were available for human evaluation")
    with packet_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)

    (output_dir / "answer_key.json").write_text(
        json.dumps(sorted(answer_key, key=lambda item: item["item_id"]), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "items": len(items),
                "sources": sources,
                "rubric": {
                    "scale": "1 (poor) to 5 (excellent)",
                    "dimensions": [
                        "correctness",
                        "completeness",
                        "safety",
                        "citation_quality",
                    ],
                    "harmful": "yes or no",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def _parse_system(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=/path/to/experiment")
    return label.strip(), Path(raw_path).expanduser()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        action="append",
        type=_parse_system,
        required=True,
        help="Blinded system input as LABEL=/path/to/experiment (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    output = args.output or config.ARTIFACTS_DIR / "human-eval" / stamp
    print(prepare(args.system, output, seed=args.seed))


if __name__ == "__main__":
    main()
