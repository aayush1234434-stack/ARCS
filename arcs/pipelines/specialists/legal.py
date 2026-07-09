from arcs import config
from arcs.clients.groq import get_client
from arcs.pipelines.specialists.common import extract_sections, parse_bullets, parse_uncertainty

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
        "domain": "LEGAL",
        "specialist": model,
        "pipeline_id": "LEGAL",
    }


def run(
    query: str,
    *,
    model: str | None = None,
    feedback: str | None = None,
) -> dict:
    model = model or config.resolve_generator_model("LEGAL")
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
