from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.prepare_human_eval import prepare


def _experiment(path: Path, answer: str) -> Path:
    path.mkdir()
    (path / "experiment.json").write_text(
        json.dumps(
            {
                "meta": {
                    "rows": [
                        {
                            "id": "q-1",
                            "query": "Question?",
                            "expected_domain": "GENERAL",
                            "specialist": {"answer": answer},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prepare_human_eval_blinds_and_randomizes_system_labels(tmp_path: Path):
    a = _experiment(tmp_path / "a", "Answer A")
    b = _experiment(tmp_path / "b", "Answer B")
    output = prepare([("naive", a), ("arcs", b)], tmp_path / "packet", seed=7)

    with (output / "review_packet.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all("system" not in row for row in rows)
    assert {row["answer"] for row in rows} == {"Answer A", "Answer B"}

    key = json.loads((output / "answer_key.json").read_text(encoding="utf-8"))
    assert {item["system"] for item in key} == {"naive", "arcs"}
