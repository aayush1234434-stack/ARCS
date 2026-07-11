"""Apply a DSPy sidecar prompt to a source module's SYSTEM_PROMPT.

Sidecar files (from optimize_*.py) include a comment header; this script strips
it and replaces the triple-quoted SYSTEM_PROMPT assignment in the target file.
Creates ``<target>.bak`` before writing.
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_SYSTEM_PROMPT_RE = re.compile(
    r'^(?P<prefix>\s*SYSTEM_PROMPT\s*=\s*)"""(?P<body>.*?)"""',
    re.DOTALL | re.MULTILINE,
)


def _read_sidecar(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Sidecar not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    body_lines: list[str] = []
    in_body = False
    for line in lines:
        if not in_body and (not line.strip() or line.lstrip().startswith("#")):
            continue
        in_body = True
        body_lines.append(line)

    text = "\n".join(body_lines).strip("\n")
    if not text.strip():
        raise ValueError(f"No prompt text found in sidecar (after comment header): {path}")
    if '"""' in text:
        raise ValueError(
            "Sidecar prompt contains triple quotes; edit manually or escape before apply."
        )
    return text


def _extract_system_prompt(source: str) -> tuple[str, str, str]:
    match = _SYSTEM_PROMPT_RE.search(source)
    if not match:
        raise ValueError("Target file has no SYSTEM_PROMPT = \"\"\"...\"\"\" assignment")
    return match.group("prefix"), match.group("body"), match.group(0)


def _replace_system_prompt(source: str, new_body: str) -> str:
    prefix, _old_body, matched = _extract_system_prompt(source)
    replacement = f'{prefix}"""{new_body}"""'
    return source.replace(matched, replacement, 1)


def apply_sidecar(
    *,
    prompt_path: Path,
    target_path: Path,
    dry_run: bool = False,
) -> None:
    sidecar_text = _read_sidecar(prompt_path)
    target_path = target_path.resolve()
    original = target_path.read_text(encoding="utf-8")
    updated = _replace_system_prompt(original, sidecar_text)

    if original == updated:
        print("No change: sidecar prompt matches current SYSTEM_PROMPT.", file=sys.stderr)
        return

    if dry_run:
        print(f"--- dry-run: would update {target_path}", file=sys.stderr)
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(target_path),
            tofile=f"{target_path} (with sidecar)",
        )
        sys.stdout.writelines(diff)
        return

    backup = target_path.with_suffix(target_path.suffix + ".bak")
    shutil.copy2(target_path, backup)
    target_path.write_text(updated, encoding="utf-8")
    print(f"Backup  → {backup}", file=sys.stderr)
    print(f"Updated → {target_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace SYSTEM_PROMPT in a target module with the body of a DSPy "
            "sidecar file (comment header stripped). Writes <target>.bak first."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help="Sidecar prompt file (e.g. artifacts/prompts/coding_optimized.txt)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Source module containing SYSTEM_PROMPT (e.g. arcs/pipelines/specialists/coding.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print unified diff; do not write files",
    )
    args = parser.parse_args()

    prompt_path = args.prompt if args.prompt.is_absolute() else _ROOT / args.prompt
    target_path = args.target if args.target.is_absolute() else _ROOT / args.target

    try:
        apply_sidecar(
            prompt_path=prompt_path,
            target_path=target_path,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
