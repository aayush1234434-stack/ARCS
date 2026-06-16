import os
from groq import Groq
from dotenv import load_dotenv

# Initialize client with API key from environment variable
load_dotenv()  # Load environment variables from .env file
client = Groq()

SYSTEM_PROMPT = """You are an expert coding specialist with deep knowledge of algorithms, data structures, and software engineering best practices.

When given a coding problem:
1. Provide a complete, working solution
2. Use clean, readable code with comments where needed
3. Briefly explain your approach and why you chose it
4. State the time and space complexity
5. Handle edge cases

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
<list any claims or parts of the solution you are not fully confident about, or 'None' if fully confident>"""

query = """Write a Python function that takes a list of integers and returns the two numbers that add up to a specific target."""


def parse_response(raw: str) -> dict:
    sections = {}
    keys = ["SOLUTION", "EXPLANATION", "COMPLEXITY", "EDGE CASES", "UNCERTAINTY"]
    for i, key in enumerate(keys):
        start = raw.find(f"{key}:")
        if start == -1:
            sections[key.lower()] = ""
            continue
        start += len(f"{key}:")
        end = len(raw)
        for next_key in keys[i+1:]:
            pos = raw.find(f"{next_key}:", start)
            if pos != -1:
                end = pos
                break
        sections[key.lower()] = raw[start:end].strip()

    uncertainty_raw = sections.get("uncertainty", "None")
    uncertainty = [] if uncertainty_raw.strip().lower().startswith("none") else [
        u.strip() for u in uncertainty_raw.split("\n") if u.strip()
    ]

    return {
        "answer": sections.get("solution", ""),
        "explanation": sections.get("explanation", ""),
        "complexity": sections.get("complexity", ""),
        "edge_cases": sections.get("edge cases", ""),
        "uncertainty": uncertainty,
        "domain": "CODING",
        "specialist": "llama-3.3-70b-versatile"
    }


def run(query: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )
    raw = response.choices[0].message.content
    return parse_response(raw)

