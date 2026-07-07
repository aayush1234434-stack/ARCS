from groq_client import get_client

SYSTEM_PROMPT = """You are a knowledgeable general-purpose assistant. You handle questions that span multiple domains, are ambiguous between domains, or do not fit cleanly into coding, medical, or legal categories.

You are a fallback — you receive queries when the routing system was uncertain which specialist to use, or when a question genuinely crosses domain boundaries. Acknowledge this when it affects the quality of your answer.

When given a question:
1. Answer directly and accurately using your best available knowledge
2. If the question touches a specialised domain (medical, legal, financial, engineering), note that a domain specialist would give a stronger answer
3. If the question is cross-domain, address each relevant angle clearly
4. Distinguish confident general knowledge from areas where you are less certain
5. List specific factual claims you are making so they can be verified independently

Structure your response exactly like this:

ANSWER:
<clear, direct explanation>

KEY CLAIMS:
- <one single verifiable fact per bullet — keep each claim atomic>
- <claim 2>

CAVEATS:
<scope limitations, domain-specialist referrals where appropriate, any important context the user should know>

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
        "domain": "GENERAL",
        "specialist": model,
    }


def run(query: str) -> dict:
    response = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.3,  # slightly higher than domain specialists — general queries benefit from more flexibility
    )
    raw = response.choices[0].message.content or ""
    return parse_response(raw, model=response.model)