from groq_client import get_client

SYSTEM_PROMPT = """You are an expert legal information specialist. You provide clear, accurate legal information for educational purposes only.

You are NOT a lawyer and cannot provide legal advice, represent anyone, or replace qualified legal counsel. Always encourage users to consult a licensed attorney for decisions affecting their legal rights.

When given a legal question:
1. Answer directly using established law, common legal principles, and jurisdiction-general guidance
2. Flag when the answer is jurisdiction-specific and may differ by country, state, or region
3. Distinguish between what the law says and what commonly happens in practice
4. Note when the question requires review of specific documents or facts only an attorney can assess
5. List specific factual legal claims you are making so they can be verified independently

Structure your response exactly like this:

ANSWER:
<clear, plain-language explanation>

KEY CLAIMS:
- <one single verifiable legal fact per bullet — keep each claim atomic, e.g. "In the US, non-compete agreements are unenforceable in California" not "non-competes vary by state and context">
- <claim 2>

CAVEATS:
<jurisdiction warnings, document-specific limitations, when to consult an attorney urgently or routinely>

UNCERTAINTY:
- <specific claim or jurisdiction you are not fully confident about>
- <or write exactly: None>"""


def _parse_bullets(text: str) -> list[str]:
    """Parse bullet lists in -, •, *, or 1. / (a) formats."""
    import re
    items = []
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


def _parse_uncertainty(raw: str) -> list[str]:
    """
    Return an empty list when the specialist is fully confident,
    a list of specific uncertain claims otherwise.
    Handles both bullet-list and prose responses.
    """
    raw = raw.strip()
    if not raw or raw.lower() == "none" or raw.lower().startswith("none"):
        return []

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    has_bullets = any(
        l.startswith(("-", "•", "*")) or (len(l) > 2 and l[0].isdigit() and l[1] in ".)")
        for l in lines
    )

    if has_bullets:
        parsed = _parse_bullets(raw)
        if len(parsed) == 1 and parsed[0].lower() == "none":
            return []
        return parsed

    return [raw]


def parse_response(raw: str, model: str) -> dict:
    sections: dict[str, str] = {}
    keys = ["ANSWER", "KEY CLAIMS", "CAVEATS", "UNCERTAINTY"]

    for i, key in enumerate(keys):
        start = raw.find(f"{key}:")
        if start == -1:
            sections[key.lower()] = ""
            continue
        start += len(f"{key}:")
        end = len(raw)
        for next_key in keys[i + 1:]:
            pos = raw.find(f"{next_key}:", start)
            if pos != -1:
                end = pos
                break
        sections[key.lower()] = raw[start:end].strip()

    uncertainty = _parse_uncertainty(sections.get("uncertainty", "None"))
    claims = _parse_bullets(sections.get("key claims", ""))

    return {
        "answer": sections.get("answer", ""),
        "claims": claims,
        "caveats": sections.get("caveats", ""),
        "specialist_uncertainty": len(uncertainty) > 0,
        "specialist_uncertainty_claims": uncertainty,
        "domain": "LEGAL",
        "specialist": model,
    }


def run(query: str) -> dict:
    response = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    return parse_response(raw, model=response.model)