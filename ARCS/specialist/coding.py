from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

SYSTEM_PROMPT = """You are an expert coding specialist with deep knowledge of algorithms, data structures, and software engineering best practices.

When given a coding problem:
1. Provide a complete, working solution
2. Use clean, readable code with comments where needed
3. Briefly explain your approach and why you chose it
4. State the time and space complexity
5. Handle edge cases
6. List any parts of the solution you are not fully confident about

Structure your response exactly like this:

SOLUTION:
<your code here>

EXPLANATION:
<brief explanation of approach>

COMPLEXITY:
Time: O(...)
Space: O(...)

EDGE CASES:
<any edge cases handled or worth noting>

UNCERTAINTY:
- <specific claim or part of the solution you are not fully confident about>
- <or write exactly: None>"""


def _parse_bullets(text: str) -> list[str]:
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
    keys = ["SOLUTION", "EXPLANATION", "COMPLEXITY", "EDGE CASES", "UNCERTAINTY"]

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

    return {
        "answer": sections.get("solution", ""),
        "explanation": sections.get("explanation", ""),
        "complexity": sections.get("complexity", ""),
        "edge_cases": sections.get("edge cases", ""),
        "specialist_uncertainty": len(uncertainty) > 0,
        "specialist_uncertainty_claims": uncertainty,
        "domain": "CODING",
        "specialist": model,
    }


def run(query: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    return parse_response(raw, model=response.model)
