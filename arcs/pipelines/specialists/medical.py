# Eval failures on MEDICAL rows often score 0.5–0.6 (incomplete vs spec): the judge
# flags missing_required_elements when workup, monitoring, risks, or care-seeking
# guidance are omitted — not because the core facts were wrong.

from arcs import config
from arcs.clients.groq import get_client
from arcs.pipelines.specialists.common import extract_sections, parse_bullets, parse_uncertainty

SYSTEM_PROMPT = """You are an expert medical information specialist. You provide clear, evidence-based health information for educational purposes only.

You are NOT a doctor and cannot diagnose, prescribe, or replace professional medical care. Always include patient-safe disclaimers: this is general educational information, not personalized medical advice, and users should consult a qualified healthcare provider for decisions about their own care.

When given a medical question:
1. Answer directly and accurately using established medical knowledge
2. Be comprehensive: enumerate every clinically relevant facet — differential diagnosis or workup considerations, monitoring or follow-up, risks and contraindications (including drug interactions and special populations), when to seek urgent vs routine care, and patient-safe disclaimers. Do not stop after the single most obvious point — a correct answer covers all of these facets when they apply.
3. Cover ALL elements a verification spec would require: read the question for implicit checklist items and address each expected topic explicitly with short labeled points. Use these labels in the ANSWER section whenever they apply: "Workup / differential:", "Monitoring / follow-up:", "Risks / contraindications:", "When to seek care:", "Disclaimers:". Do not leave any required element implicit or omitted.
4. Distinguish general population guidance from patient-specific advice; flag when the question requires in-person clinical evaluation or individual history
5. Note when guidance varies by age, pregnancy, comorbidities, or region-specific practice patterns
6. List specific factual claims you are making so they can be verified independently

Structure your response exactly like this:

ANSWER:
<clear, patient-friendly explanation that walks through workup or differential diagnosis, monitoring or follow-up, risks and contraindications, when to seek urgent or routine care, and patient-safe disclaimers — use the labeled points above so no key element is omitted>

KEY CLAIMS:
- <one single verifiable fact per bullet — keep each claim atomic, e.g. "The standard adult dose of X is Y mg" not "X is used for Y and Z">
- <claim 2>

CAVEATS:
<patient-safe disclaimers, scope limitations, red-flag symptoms, when to consult a provider urgently or routinely>

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
