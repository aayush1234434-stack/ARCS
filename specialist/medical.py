from groq_client import get_client

SYSTEM_PROMPT = """You are an expert medical information specialist. You provide clear, evidence-based health information for educational purposes only.

You are NOT a doctor and cannot diagnose, prescribe, or replace professional medical care. Always encourage users to consult a qualified healthcare provider for personal medical decisions.

When given a medical question:
1. Answer directly and accurately using established medical knowledge
2. Distinguish general population guidance from patient-specific advice
3. Note relevant contraindications, drug interactions, or special populations when applicable
4. Flag when the question requires in-person clinical evaluation
5. List specific factual claims you are making so they can be verified independently

Structure your response exactly like this:

ANSWER:
<clear, patient-friendly explanation>

KEY CLAIMS:
- <one single verifiable fact per bullet — keep each claim atomic, e.g. "The standard adult dose of X is Y mg" not "X is used for Y and Z">
- <claim 2>

CAVEATS:
<warnings, limitations, when to seek urgent or routine care>

UNCERTAINTY:
- <specific claim or topic you are not fully confident about>
- <or write exactly: None>"""


def _parse_bullets(text: str) -> list[str]:
    """Parse bullet lists in -, •, *, or 1. / (a) formats."""
    import re
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading bullet or numbering
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

    # Check whether any line looks like a bullet
    has_bullets = any(
        l.startswith(("-", "•", "*")) or (len(l) > 2 and l[0].isdigit() and l[1] in ".)") 
        for l in lines
    )

    if has_bullets:
        parsed = _parse_bullets(raw)
        # After stripping bullets, a lone "None" still means no uncertainty
        if len(parsed) == 1 and parsed[0].lower() == "none":
            return []
        return parsed

    # Prose uncertainty — wrap as a single item so attribution can use it
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
        # Attribution engine reads these two fields directly
        "specialist_uncertainty": len(uncertainty) > 0,
        "specialist_uncertainty_claims": uncertainty,
        "domain": "MEDICAL",
        "specialist": model,  # pulled from response, not hardcoded
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



