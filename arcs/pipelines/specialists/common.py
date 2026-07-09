"""Shared parsing helpers for specialist structured responses."""

from __future__ import annotations

import re


def parse_bullets(text: str) -> list[str]:
    """Parse bullet lists in -, •, *, or 1. / (a) formats."""
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-•*]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = re.sub(r"^\([a-zA-Z]\)\s+", "", line)
        if line:
            items.append(line)
    return items


def parse_uncertainty(raw: str) -> list[str]:
    """
    Return an empty list when the specialist is fully confident,
    a list of specific uncertain claims otherwise.
    """
    raw = raw.strip()
    if not raw or raw.lower() == "none" or raw.lower().startswith("none"):
        return []

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    has_bullets = any(
        line.startswith(("-", "•", "*"))
        or (len(line) > 2 and line[0].isdigit() and line[1] in ".)")
        for line in lines
    )

    if has_bullets:
        parsed = parse_bullets(raw)
        if len(parsed) == 1 and parsed[0].lower() == "none":
            return []
        return parsed

    return [raw]


def extract_sections(raw: str, keys: list[str]) -> dict[str, str]:
    """Split a labeled-section response into a dict keyed by lowercased labels."""
    sections: dict[str, str] = {}

    for index, key in enumerate(keys):
        start = raw.find(f"{key}:")
        if start == -1:
            sections[key.lower()] = ""
            continue
        start += len(f"{key}:")
        end = len(raw)
        for next_key in keys[index + 1 :]:
            pos = raw.find(f"{next_key}:", start)
            if pos != -1:
                end = pos
                break
        sections[key.lower()] = raw[start:end].strip()

    return sections


def extract_code_block(text: str) -> str:
    """Pull the first fenced Python/code block, or return stripped text."""
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (fence.group(1) if fence else text).strip()
