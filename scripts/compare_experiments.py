"""CLI: compare two saved Phase 2 experiment runs.

Exit code is always 0 on successful comparison (not a quality gate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.eval.compare import diff_experiments, format_diff
from arcs.eval.experiments import latest_experiment, load_experiment


def _resolve_experiment_path(raw: str | Path, *, experiments_dir: Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    # Treat as run_id under experiments dir.
    candidate = experiments_dir / str(raw)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"experiment not found: {raw}\n"
        f"Tried: {path.resolve() if path.is_absolute() else path}\n"
        f"Tried: {candidate}"
    )


def _list_experiments(experiments_dir: Path, *, limit: int = 20) -> list[Path]:
    if not experiments_dir.exists():
        return []
    runs = [
        p
        for p in experiments_dir.iterdir()
        if p.is_dir() and (p / "experiment.json").exists()
    ]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs[:limit]


def _print_list(experiments_dir: Path, *, limit: int) -> None:
    runs = _list_experiments(experiments_dir, limit=limit)
    if not runs:
        print(f"No experiments under {experiments_dir}", file=sys.stderr)
        return
    print(f"Recent experiments in {experiments_dir}:", file=sys.stderr)
    for path in runs:
        try:
            data = load_experiment(path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"  {path.name}  (unreadable: {exc})", file=sys.stderr)
            continue
        kind = data.get("kind") or (
            "pipeline" if isinstance(data.get("pipeline"), dict) else "?"
        )
        name = data.get("name") or "?"
        meta = data.get("meta") or {}
        created = meta.get("created_at") or ""
        print(
            f"  {path.name}  name={name}  kind={kind}"
            + (f"  created={created}" if created else ""),
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diff two saved experiment runs (B − A). Paths may be experiment.json "
            "files, run directories, or run_id names under artifacts/experiments/."
        ),
    )
    parser.add_argument(
        "path_a",
        nargs="?",
        help="Experiment A (baseline)",
    )
    parser.add_argument(
        "path_b",
        nargs="?",
        help="Experiment B (candidate)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest experiment as A (requires --vs)",
    )
    parser.add_argument(
        "--vs",
        dest="vs",
        default=None,
        help="Experiment B when using --latest",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent experiments and exit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max runs to show with --list (default: 20)",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help=f"Experiments root (default: {config.EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable diff dict to stdout",
    )
    args = parser.parse_args()

    experiments_dir = args.experiments_dir or config.EXPERIMENTS_DIR

    if args.list:
        _print_list(experiments_dir, limit=args.limit)
        return

    if args.latest:
        if not args.vs:
            print("Error: --latest requires --vs <path_b>", file=sys.stderr)
            sys.exit(1)
        latest = latest_experiment(experiments_dir)
        if latest is None:
            print(f"Error: no experiments under {experiments_dir}", file=sys.stderr)
            sys.exit(1)
        path_a = latest
        path_b_raw = args.vs
    else:
        if not args.path_a or not args.path_b:
            parser.print_help()
            print(
                "\nExamples:\n"
                "  python scripts/compare_experiments.py --list\n"
                "  python scripts/compare_experiments.py <run_a> <run_b>\n"
                "  python scripts/compare_experiments.py --latest --vs <run_b>",
                file=sys.stderr,
            )
            sys.exit(1)
        path_a = args.path_a
        path_b_raw = args.path_b

    try:
        resolved_a = _resolve_experiment_path(path_a, experiments_dir=experiments_dir)
        resolved_b = _resolve_experiment_path(path_b_raw, experiments_dir=experiments_dir)
        exp_a = load_experiment(resolved_a)
        exp_b = load_experiment(resolved_b)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"Error: failed to load experiment: {exc}", file=sys.stderr)
        sys.exit(1)

    # Ensure run_id is present for display when loading from a folder name.
    for exp, resolved in ((exp_a, resolved_a), (exp_b, resolved_b)):
        meta = exp.setdefault("meta", {})
        if isinstance(meta, dict) and not meta.get("run_id"):
            folder = resolved if resolved.is_dir() else resolved.parent
            meta["run_id"] = folder.name

    diff = diff_experiments(exp_a, exp_b)
    print(format_diff(diff), file=sys.stderr)
    if args.json:
        print(json.dumps(diff, indent=2, default=str))


if __name__ == "__main__":
    main()
