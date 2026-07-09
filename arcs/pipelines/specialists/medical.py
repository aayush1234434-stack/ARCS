from arcs import config
from arcs.clients.groq import get_client
from arcs.pipelines.specialists.common import extract_sections, parse_bullets, parse_uncertainty

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


def parse_response(raw: str, model: str) -> dict:
    sections = extract_sections(
        raw,
        ["ANSWER", "KEY CLAIMS", "CAVEATS", "UNCERTAINTY"],
    )
    uncertainty = parse_uncertainty(sections.get("uncertainty", "None"))
    claims = parse_bullets(sections.get("key claims", ""))

    return {
        "answer": sections.get("answer", ""),
        "claims": claims,
        "caveats": sections.get("caveats", ""),
        "specialist_uncertainty": len(uncertainty) > 0,
        "specialist_uncertainty_claims": uncertainty,
        "domain": "MEDICAL",
        "specialist": model,
        "pipeline_id": "MEDICAL",
    }


def run(
    query: str,
    *,
    model: str | None = None,
    feedback: str | None = None,
) -> dict:
    model = model or config.resolve_generator_model("MEDICAL")
    user_content = query
    if feedback:
        user_content = f"{query}\n\nADDITIONAL CONTEXT:\n{feedback.strip()}"

    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    return parse_response(raw, model=response.model)
