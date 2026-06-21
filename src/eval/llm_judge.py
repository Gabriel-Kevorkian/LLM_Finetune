"""
LLM-as-Judge scoring using GPT-4o-mini.

WHAT IT DOES:
    For each (question, reference, prediction) triple, we ask GPT-4o-mini
    to rate the prediction's quality on a 1-5 scale relative to the
    reference answer. This catches cases that ROUGE misses:

        Reference:  "docker ps -a lists every container including stopped ones"
        Prediction: "Use 'docker ps --all' to see all containers, even stopped"

    ROUGE-1 here is low (different wording) but the answer is essentially
    correct. A human judge — or GPT-4o-mini — would rate it 4 or 5.

WHY GPT-4o-MINI:
    - ~30x cheaper than GPT-4o ($0.15 vs $5.00 per million input tokens).
    - Smart enough to grade short technical answers reliably.
    - A full 50-question eval run costs roughly $0.01 of API spend.

WHY TEMPERATURE = 0:
    Judging must be reproducible. Same inputs -> same score every run.

RUBRIC DESIGN:
    A vague prompt ("rate from 1 to 5") gives noisy ratings. We define
    each score level explicitly so the judge clusters around the right
    bucket consistently.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from openai import OpenAI

# Load .env so OPENAI_API_KEY is available as an env var. Idempotent —
# safe to call from multiple modules.
load_dotenv()


_JUDGE_PROMPT = """You are grading a candidate answer to a Docker question \
against a reference answer.

QUESTION:
{question}

REFERENCE ANSWER (the correct one, from official Docker docs):
{reference}

CANDIDATE ANSWER (to be graded):
{candidate}

Grade the candidate from 1 to 5 using this rubric:
5 = Fully correct AND complete. Covers the same key facts as the reference \
even if phrased differently.
4 = Correct but missing a minor detail, OR includes one small inaccuracy \
that doesn't mislead.
3 = Partially correct. Gets the gist but misses an important point or \
includes a meaningful error.
2 = Mostly wrong but contains some relevant content.
1 = Wrong, off-topic, refuses to answer, or is empty.

Reply with ONLY a single integer from 1 to 5. No other text."""


def judge_pair(
    question: str,
    reference: str,
    candidate: str,
    *,
    model: str = "gpt-4o-mini",
    client: OpenAI | None = None,
) -> int:
    """Return an integer score from 1 to 5.

    Falls back to 1 (worst score) if parsing fails — better to under-credit
    than to silently insert wrong scores into our results.
    """
    if client is None:
        # OpenAI() reads OPENAI_API_KEY from env automatically.
        client = OpenAI()

    # Guard against empty candidate strings (model refused, generation
    # hit max tokens with nothing useful, etc.). These are real failures
    # and should score 1 without burning an API call.
    if not candidate.strip():
        return 1

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    question=question,
                    reference=reference,
                    candidate=candidate,
                ),
            }
        ],
        temperature=0.0,
        max_tokens=4,   # we only need a single digit
    )

    raw = response.choices[0].message.content or ""
    return _parse_score(raw)


def _parse_score(raw: str) -> int:
    """Extract an integer 1-5 from the model's reply.

    Robust to leading/trailing whitespace, periods, and the occasional
    'Score: 4' formatting that the model sometimes adds despite our prompt.
    """
    match = re.search(r"\b([1-5])\b", raw)
    if match is None:
        # Defensive: if we can't parse, treat as worst score so the
        # bad output is visible in results rather than silently averaged.
        return 1
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Smoke test (uses 1 API call, costs < $0.001):
#     python -m src.eval.llm_judge
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    score = judge_pair(
        question="How do you list all containers, including stopped ones?",
        reference="docker ps -a. The -a flag includes stopped containers.",
        candidate="Use 'docker ps --all' to list every container including stopped ones.",
    )
    print(f"LLM-judge score (should be 4 or 5): {score}")
