from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

# Deliberately different model family from the Llama-based specialists.
# Gemma is Google's architecture — correlated blind spots with Meta's Llama are minimal.
MODEL = "gemma2-9b-it"

SYSTEM_PROMPT = """You are a specification generator. Your job is not to answer questions — it is to define what a correct answer must look like.

You will receive a user query. You will output a structured specification that a judge can use to evaluate whether any given answer to that query is correct, complete, and appropriate.

Be precise and testable. Every criterion you list must be something a judge can check against an answer without needing domain expertise — it should be checkable by reading.

Structure your response exactly like this:

INTENT:
<one sentence: what is the user actually trying to accomplish>

REQUIRED ELEMENTS:
- <specific thing that must be present in a correct answer>
- <required element 2>

CORRECTNESS CRITERIA:
- <specific checkable condition that a correct answer satisfies>
- <criterion 2>

DISQUALIFYING CONDITIONS:
- <specific thing that, if present, makes the answer wrong or incomplete regardless of anything else>
- <disqualifying condition 2>

SCOPE:
<what the answer should and should not cover — flag if the question is ambiguous or cross-domain>"""


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


def parse_response(raw: str, model: str) -> dict:
    sections: dict[str, str] = {}
    keys = ["INTENT", "REQUIRED ELEMENTS", "CORRECTNESS CRITERIA", "DISQUALIFYING CONDITIONS", "SCOPE"]

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

    return {
        "intent":                   sections.get("intent", ""),
        "required_elements":        _parse_bullets(sections.get("required elements", "")),
        "correctness_criteria":     _parse_bullets(sections.get("correctness criteria", "")),
        "disqualifying_conditions": _parse_bullets(sections.get("disqualifying conditions", "")),
        "scope":                    sections.get("scope", ""),
        "model":                    model,
    }


def run(query: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or ""
    return parse_response(raw, model=response.model)