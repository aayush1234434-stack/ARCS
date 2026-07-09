"""
Pure experiment comparison helpers for Phase 2.

No I/O — ``diff_experiments(a, b)`` returns a JSON-serializable delta dict.
"""

from __future__ import annotations

from typing import Any

from arcs.eval.metrics import VALID_DOMAINS


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(a: Any, b: Any) -> dict[str, float | None]:
    """Return ``{a, b, delta}`` where delta = b - a (None if either missing)."""
    left = _as_float(a)
    right = _as_float(b)
    return {
        "a": left,
        "b": right,
        "delta": (None if left is None or right is None else right - left),
    }


def _infer_kind(experiment: dict[str, Any]) -> str:
    kind = experiment.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip().lower()
    if isinstance(experiment.get("metrics"), dict) and (
        "macro_f1" in experiment["metrics"] or "per_class" in experiment["metrics"]
    ):
        return "router"
    if isinstance(experiment.get("pipeline"), dict):
        return "pipeline"
    if isinstance(experiment.get("router"), dict):
        return "pipeline"
    return "unknown"


def _router_metric_block(experiment: dict[str, Any]) -> dict[str, Any]:
    """Pull accuracy / F1 from ``metrics`` (router-eval) or ``router`` (pipeline-eval)."""
    metrics = experiment.get("metrics") if isinstance(experiment.get("metrics"), dict) else {}
    router = experiment.get("router") if isinstance(experiment.get("router"), dict) else {}

    accuracy = metrics.get("accuracy", router.get("accuracy"))
    macro_f1 = metrics.get("macro_f1", router.get("macro_f1"))
    balanced = metrics.get("balanced_accuracy", router.get("balanced_accuracy"))

    per_class_f1: dict[str, float | None] = {}
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
    per_domain = (
        router.get("per_domain_accuracy")
        if isinstance(router.get("per_domain_accuracy"), dict)
        else {}
    )
    for domain in VALID_DOMAINS:
        f1 = None
        if domain in per_class and isinstance(per_class[domain], dict):
            f1 = _as_float(per_class[domain].get("f1"))
        if f1 is None:
            f1 = _as_float(per_domain.get(domain))
        per_class_f1[domain] = f1

    return {
        "accuracy": _as_float(accuracy),
        "macro_f1": _as_float(macro_f1),
        "balanced_accuracy": _as_float(balanced),
        "per_class_f1": per_class_f1,
    }


def _pipeline_metric_block(experiment: dict[str, Any]) -> dict[str, Any]:
    pipeline = experiment.get("pipeline") if isinstance(experiment.get("pipeline"), dict) else {}
    router = experiment.get("router") if isinstance(experiment.get("router"), dict) else {}
    rates = pipeline.get("status_rates") if isinstance(pipeline.get("status_rates"), dict) else {}
    latency = pipeline.get("latency_ms") if isinstance(pipeline.get("latency_ms"), dict) else {}
    total = latency.get("total_ms") if isinstance(latency.get("total_ms"), dict) else {}

    per_domain_pass: dict[str, float | None] = {}
    per_domain = pipeline.get("per_domain") if isinstance(pipeline.get("per_domain"), dict) else {}
    for domain in VALID_DOMAINS:
        bucket = per_domain.get(domain)
        if isinstance(bucket, dict):
            per_domain_pass[domain] = _as_float(bucket.get("pass_rate"))
        else:
            per_domain_pass[domain] = None

    return {
        "pass_rate": _as_float(rates.get("PASS")),
        "fail_rate": _as_float(rates.get("FAIL")),
        "error_rate": _as_float(rates.get("ERROR")),
        "router_accuracy": _as_float(router.get("accuracy")),
        "latency_mean_ms": _as_float(total.get("mean")),
        "latency_p95_ms": _as_float(total.get("p95")),
        "per_domain_pass_rate": per_domain_pass,
        "n": pipeline.get("n"),
    }


def diff_experiments(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two experiment dicts. ``delta`` is always B − A.

    Higher is better for accuracy / F1 / PASS rate.
    Lower is better for latency (still reported as B − A; callers interpret sign).
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise TypeError("both experiments must be dicts")

    kind_a = _infer_kind(a)
    kind_b = _infer_kind(b)
    kind = kind_a if kind_a == kind_b else "mixed"

    meta_a = a.get("meta") if isinstance(a.get("meta"), dict) else {}
    meta_b = b.get("meta") if isinstance(b.get("meta"), dict) else {}

    result: dict[str, Any] = {
        "kind": kind,
        "a": {
            "name": a.get("name"),
            "run_id": meta_a.get("run_id"),
            "kind": kind_a,
        },
        "b": {
            "name": b.get("name"),
            "run_id": meta_b.get("run_id"),
            "kind": kind_b,
        },
        "router": None,
        "pipeline": None,
    }

    # Router-style metrics (also useful for pipeline experiments' router block).
    ra = _router_metric_block(a)
    rb = _router_metric_block(b)
    if any(v is not None for v in (ra["accuracy"], ra["macro_f1"], rb["accuracy"], rb["macro_f1"])):
        per_class_delta = {
            domain: _delta(ra["per_class_f1"].get(domain), rb["per_class_f1"].get(domain))
            for domain in VALID_DOMAINS
        }
        result["router"] = {
            "accuracy": _delta(ra["accuracy"], rb["accuracy"]),
            "macro_f1": _delta(ra["macro_f1"], rb["macro_f1"]),
            "balanced_accuracy": _delta(ra["balanced_accuracy"], rb["balanced_accuracy"]),
            "per_class_f1": per_class_delta,
        }

    # Pipeline-style metrics.
    pa = _pipeline_metric_block(a)
    pb = _pipeline_metric_block(b)
    if any(
        v is not None
        for v in (
            pa["pass_rate"],
            pb["pass_rate"],
            pa["latency_mean_ms"],
            pb["latency_mean_ms"],
            pa["n"],
            pb["n"],
        )
    ):
        result["pipeline"] = {
            "pass_rate": _delta(pa["pass_rate"], pb["pass_rate"]),
            "fail_rate": _delta(pa["fail_rate"], pb["fail_rate"]),
            "error_rate": _delta(pa["error_rate"], pb["error_rate"]),
            "router_accuracy": _delta(pa["router_accuracy"], pb["router_accuracy"]),
            "latency_mean_ms": _delta(pa["latency_mean_ms"], pb["latency_mean_ms"]),
            "latency_p95_ms": _delta(pa["latency_p95_ms"], pb["latency_p95_ms"]),
            "per_domain_pass_rate": {
                domain: _delta(
                    pa["per_domain_pass_rate"].get(domain),
                    pb["per_domain_pass_rate"].get(domain),
                )
                for domain in VALID_DOMAINS
            },
            "n": {"a": pa["n"], "b": pb["n"]},
        }

    return result


def format_diff(diff: dict[str, Any], *, higher_is_better_latency: bool = False) -> str:
    """Human-readable multi-line summary. ``+`` = improvement, ``-`` = regression."""

    def _fmt(value: float | None, *, digits: int = 4) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{digits}f}"

    def _signed(delta: float | None, *, digits: int = 4, invert: bool = False) -> str:
        if delta is None:
            return "n/a"
        # Flip so that a positive display value always means "B is better".
        display = -delta if invert else delta
        eps = 10 ** (-(digits + 1))
        if abs(display) < eps:
            return f"={abs(display):.{digits}f}"
        marker = "+" if display > 0 else "-"
        return f"{marker}{abs(display):.{digits}f}"

    lines: list[str] = []
    a = diff.get("a") or {}
    b = diff.get("b") or {}
    lines.append("Experiment comparison (B − A)")
    lines.append(
        f"  A: {a.get('name') or '?'}  ({a.get('run_id') or a.get('kind') or '?'})"
    )
    lines.append(
        f"  B: {b.get('name') or '?'}  ({b.get('run_id') or b.get('kind') or '?'})"
    )
    lines.append(f"  kind: {diff.get('kind')}")
    lines.append("")

    router = diff.get("router")
    if isinstance(router, dict):
        lines.append("Router metrics")
        lines.append("-" * 48)
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            block = router.get(key) or {}
            lines.append(
                f"  {key:20s}  A={_fmt(block.get('a'))}  "
                f"B={_fmt(block.get('b'))}  {_signed(block.get('delta'))}"
            )
        per = router.get("per_class_f1") or {}
        if isinstance(per, dict):
            lines.append("  per-class F1:")
            for domain in VALID_DOMAINS:
                block = per.get(domain) or {}
                lines.append(
                    f"    {domain:8s}  A={_fmt(block.get('a'))}  "
                    f"B={_fmt(block.get('b'))}  {_signed(block.get('delta'))}"
                )
        lines.append("")

    pipeline = diff.get("pipeline")
    if isinstance(pipeline, dict):
        lines.append("Pipeline metrics")
        lines.append("-" * 48)
        for key, invert in (
            ("pass_rate", False),
            ("router_accuracy", False),
            ("fail_rate", True),
            ("error_rate", True),
        ):
            block = pipeline.get(key) or {}
            if block.get("a") is None and block.get("b") is None:
                continue
            lines.append(
                f"  {key:20s}  A={_fmt(block.get('a'))}  "
                f"B={_fmt(block.get('b'))}  "
                f"{_signed(block.get('delta'), invert=invert)}"
            )
        for key in ("latency_mean_ms", "latency_p95_ms"):
            block = pipeline.get(key) or {}
            if block.get("a") is None and block.get("b") is None:
                continue
            # Lower latency is better unless caller flips the convention.
            lines.append(
                f"  {key:20s}  A={_fmt(block.get('a'), digits=1)}  "
                f"B={_fmt(block.get('b'), digits=1)}  "
                f"{_signed(block.get('delta'), digits=1, invert=not higher_is_better_latency)}"
            )
        per = pipeline.get("per_domain_pass_rate") or {}
        if isinstance(per, dict):
            lines.append("  per-domain PASS rate:")
            lines.append(f"    {'domain':8s}  {'A':>8s}  {'B':>8s}  delta")
            for domain in VALID_DOMAINS:
                block = per.get(domain) or {}
                lines.append(
                    f"    {domain:8s}  {_fmt(block.get('a')):>8s}  "
                    f"{_fmt(block.get('b')):>8s}  {_signed(block.get('delta'))}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
