"""
Pure evaluation metrics for ARCS Phase 2 experiments.

No LLM calls, no I/O — deterministic aggregation only.
"""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any

VALID_DOMAINS = ("CODING", "MEDICAL", "LEGAL", "GENERAL")


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> dict[str, float | int | None]:
    """Return a Wilson score confidence interval for a binomial proportion.

    Wilson intervals remain useful for small samples where the usual normal
    approximation is misleading. The default ``z`` produces a 95% interval.
    """
    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise TypeError("successes and total must be integers")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total")
    if total == 0:
        return {"successes": successes, "n": total, "low": None, "high": None}

    proportion = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    margin = (
        z
        * sqrt((proportion * (1 - proportion) / total) + z2 / (4 * total * total))
        / denominator
    )
    return {
        "successes": successes,
        "n": total,
        "low": round(max(0.0, center - margin), 6),
        "high": round(min(1.0, center + margin), 6),
    }


def _norm_domain(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized in VALID_DOMAINS:
        return normalized
    return normalized or None


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile on a pre-sorted non-empty list."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    # Clamp pct into [0, 100]
    pct = max(0.0, min(100.0, pct))
    index = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def _latency_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean": round(mean, 3),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
    }


def router_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute router accuracy and confusion from predicted/expected domains.

    Each row should include ``predicted_domain`` and ``expected_domain``.
    """
    if not isinstance(rows, list):
        raise TypeError(f"rows must be a list, got {type(rows).__name__}")

    total = 0
    correct = 0
    per_domain_total: Counter[str] = Counter()
    per_domain_correct: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    skipped = 0

    for row in rows:
        if not isinstance(row, dict):
            skipped += 1
            continue
        expected = _norm_domain(row.get("expected_domain"))
        predicted = _norm_domain(row.get("predicted_domain"))
        if expected is None or predicted is None:
            skipped += 1
            continue

        total += 1
        per_domain_total[expected] += 1
        confusion[(expected, predicted)] += 1
        if predicted == expected:
            correct += 1
            per_domain_correct[expected] += 1

    per_domain_accuracy = {
        domain: (
            per_domain_correct[domain] / per_domain_total[domain]
            if per_domain_total[domain]
            else None
        )
        for domain in VALID_DOMAINS
    }

    confusion_matrix = {
        expected: {
            predicted: confusion.get((expected, predicted), 0)
            for predicted in VALID_DOMAINS
        }
        for expected in VALID_DOMAINS
    }

    return {
        "n": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
        "per_domain_n": {domain: per_domain_total.get(domain, 0) for domain in VALID_DOMAINS},
        "per_domain_accuracy": per_domain_accuracy,
        "confusion": confusion_matrix,
        "skipped": skipped,
    }


def pipeline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pipeline eval rows (status, route, verification, timing).

    Expected optional fields per row:
      - ``status`` / ``error``
      - ``expected_domain``
      - ``route.domain`` or ``predicted_domain``
      - ``verification.verdict``
      - ``timing`` with ``total_ms``, ``route_ms``, ``specialist_ms``, ``verification_ms``
    """
    if not isinstance(rows, list):
        raise TypeError(f"rows must be a list, got {type(rows).__name__}")

    status_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    per_domain: dict[str, dict[str, Any]] = {
        domain: {
            "n": 0,
            "status": Counter(),
            "verdict": Counter(),
        }
        for domain in VALID_DOMAINS
    }
    unknown_domain_n = 0
    error_n = 0

    latency_buckets: dict[str, list[float]] = {
        "total_ms": [],
        "route_ms": [],
        "specialist_ms": [],
        "verification_ms": [],
    }
    usage_totals: Counter[str] = Counter()
    rows_with_usage = 0

    for row in rows:
        if not isinstance(row, dict):
            continue

        has_error = bool(row.get("error"))
        status_raw = row.get("status")
        if has_error and not status_raw:
            status = "ERROR"
        elif status_raw is None:
            status = "UNKNOWN"
        else:
            status = str(status_raw).strip().upper() or "UNKNOWN"
        if has_error:
            error_n += 1
            if status not in {"ERROR", "FAIL", "PASS", "UNKNOWN"}:
                status = "ERROR"

        status_counts[status] += 1

        verification = row.get("verification") or {}
        verdict = None
        if isinstance(verification, dict):
            verdict = verification.get("verdict")
        if verdict is None:
            verdict = row.get("verdict")
        verdict_key = str(verdict).strip().upper() if verdict is not None else "UNKNOWN"
        if not verdict_key:
            verdict_key = "UNKNOWN"
        verdict_counts[verdict_key] += 1

        expected = _norm_domain(row.get("expected_domain"))
        if expected in per_domain:
            bucket = per_domain[expected]
            bucket["n"] += 1
            bucket["status"][status] += 1
            bucket["verdict"][verdict_key] += 1
        else:
            unknown_domain_n += 1

        timing = row.get("timing") or {}
        if isinstance(timing, dict):
            for key in latency_buckets:
                value = timing.get(key)
                if value is None:
                    continue
                try:
                    latency_buckets[key].append(float(value))
                except (TypeError, ValueError):
                    continue

        usage = row.get("usage") or {}
        usage_total = usage.get("total") if isinstance(usage, dict) else {}
        if isinstance(usage_total, dict) and usage_total:
            rows_with_usage += 1
            for key in ("api_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                try:
                    value = int(usage_total.get(key, 0))
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    usage_totals[key] += value

    n = len([r for r in rows if isinstance(r, dict)])
    rates = {
        "PASS": (status_counts.get("PASS", 0) / n) if n else None,
        "FAIL": (status_counts.get("FAIL", 0) / n) if n else None,
        "UNKNOWN": (status_counts.get("UNKNOWN", 0) / n) if n else None,
        "ERROR": (error_n / n) if n else None,
    }
    completed = status_counts.get("PASS", 0) + status_counts.get("FAIL", 0)
    completed_pass_ci = wilson_interval(status_counts.get("PASS", 0), completed)

    per_domain_out: dict[str, Any] = {}
    for domain in VALID_DOMAINS:
        bucket = per_domain[domain]
        dn = bucket["n"]
        per_domain_out[domain] = {
            "n": dn,
            "status_counts": dict(bucket["status"]),
            "verdict_counts": dict(bucket["verdict"]),
            "pass_rate": (
                bucket["status"].get("PASS", 0) / dn if dn else None
            ),
        }

    return {
        "n": n,
        "status_counts": dict(status_counts),
        "status_rates": rates,
        "completed_n": completed,
        "completed_pass_rate": (
            status_counts.get("PASS", 0) / completed if completed else None
        ),
        "completed_pass_ci95": completed_pass_ci,
        "error_count": error_n,
        "verdict_counts": dict(verdict_counts),
        "per_domain": per_domain_out,
        "unknown_expected_domain_n": unknown_domain_n,
        "latency_ms": {
            key: _latency_stats(values) for key, values in latency_buckets.items()
        },
        "usage": {
            "rows_with_usage": rows_with_usage,
            "totals": dict(usage_totals),
            "mean_per_evaluated_row": {
                key: round(value / rows_with_usage, 3) if rows_with_usage else None
                for key, value in usage_totals.items()
            },
            "cost_usd": None,
            "cost_note": "Measured token counts only; provider pricing is not hard-coded.",
        },
    }


def aggregate_experiment(
    name: str,
    *,
    router: dict[str, Any] | None = None,
    pipeline: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-serializable experiment result dict."""
    return {
        "name": name,
        "router": router,
        "pipeline": pipeline,
        "meta": dict(meta) if meta else {},
    }
