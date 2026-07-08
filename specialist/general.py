from groq_client import get_client

import config
from specialist.common import extract_sections, parse_bullets, parse_uncertainty

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
        "domain": "GENERAL",
        "specialist": model,
        "pipeline_id": "GENERAL",
    }


def run(
    query: str,
    *,
    model: str | None = None,
    feedback: str | None = None,
) -> dict:
    model = model or config.resolve_generator_model("GENERAL")
    user_content = query
    if feedback:
        user_content = f"{query}\n\nADDITIONAL CONTEXT:\n{feedback.strip()}"

    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    raw = response.choices[0].message.content or ""
    return parse_response(raw, model=response.model)
