from arcs import config
from arcs.clients.groq import get_client
from arcs.pipelines.specialists.common import extract_sections, parse_bullets, parse_uncertainty

SYSTEM_PROMPT = """You are an expert coding specialist with deep knowledge of algorithms, data structures, and software engineering best practices.

When given a coding problem:
1. Provide a complete, working solution
2. Use clean, readable code with comments where needed
3. Briefly explain your approach and why you chose it
4. State the time and space complexity
5. Handle edge cases
6. List any parts of the solution you are not fully confident about

If FEEDBACK FROM PREVIOUS ATTEMPT is present in the user message, fix the specific failures described. Do not ignore runtime errors or failing assertions.

Structure your response exactly like this:

SOLUTION:
<your code here — prefer a ```python fenced block>

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


def parse_response(raw: str, model: str) -> dict:
    sections = extract_sections(
        raw,
        ["SOLUTION", "EXPLANATION", "COMPLEXITY", "EDGE CASES", "UNCERTAINTY"],
    )
    uncertainty = parse_uncertainty(sections.get("uncertainty", "None"))

    return {
        "answer": sections.get("solution", ""),
        "explanation": sections.get("explanation", ""),
        "complexity": sections.get("complexity", ""),
        "edge_cases": sections.get("edge cases", ""),
        "specialist_uncertainty": len(uncertainty) > 0,
        "specialist_uncertainty_claims": uncertainty,
        "domain": "CODING",
        "specialist": model,
        "pipeline_id": "CODING",
    }


def run(
    query: str,
    *,
    model: str | None = None,
    feedback: str | None = None,
) -> dict:
    model = model or config.resolve_generator_model("CODING")
    user_content = query
    if feedback:
        user_content = (
            f"{query}\n\n"
            f"FEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback.strip()}\n\n"
            "Revise the SOLUTION so the sandbox tests pass."
        )

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
