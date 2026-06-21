"""
LLM-as-Judge scoring using Google Gemini.

WHAT IT DOES:
    For each (question, reference, prediction) triple, we ask Gemini to
    rate the prediction's quality on a 1-5 scale relative to the
    reference answer. This catches cases that ROUGE misses:

        Reference:  "docker ps -a lists every container including stopped ones"
        Prediction: "Use 'docker ps --all' to see all containers, even stopped"

    ROUGE-1 here is low (different wording) but the answer is essentially
    correct. A human judge — or Gemini — would rate it 4 or 5.

WHY GEMINI:
    - Generous free tier (15 RPM, 1500 requests/day on Flash models).
    - A full 50-question eval run uses 50 calls — well under the daily cap.
    - Gemini 2.5 Flash is fast and accurate enough for short grading tasks.
    - The judge model only needs to compare two short strings and return
      a single digit, so a small/fast model is the right choice.

WHY TEMPERATURE = 0:
    Judging must be reproducible. Same inputs -> same score every run.
    If the judge was random, our before/after comparison would be polluted
    by judge noise rather than actual model improvement.

RUBRIC DESIGN:
    A vague prompt ("rate from 1 to 5") gives noisy ratings. We define
    each score level explicitly so the judge clusters around the right
    bucket consistently.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env so GEMINI_API_KEY is available as an env var. Idempotent —
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


# Build the client once at module load — creating it per-call wastes a few ms
# of TLS handshake on each judge request.
def _make_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a free key at "
            "https://aistudio.google.com/app/apikey and add it to your .env"
        )
    return genai.Client(api_key=api_key)


_CLIENT: genai.Client | None = None


def _client() -> genai.Client:
    """Lazy singleton — only instantiate when first needed."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _make_client()
    return _CLIENT


def judge_pair(
    question: str,
    reference: str,
    candidate: str,
    *,
    model: str = "gemini-2.5-flash",
    client: genai.Client | None = None,
) -> int:
    """Return an integer score from 1 to 5.

    Falls back to 1 (worst score) if parsing fails — better to under-credit
    than to silently insert wrong scores into our results.
    """
    # Guard against empty candidate strings (model refused, generation hit
    # max tokens with nothing useful, etc.). These are real failures and
    # should score 1 without burning an API call.
    if not candidate.strip():
        return 1

    c = client or _client()

    response = c.models.generate_content(
        model=model,
        contents=_JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            candidate=candidate,
        ),
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8,  # we only need a single digit
            # IMPORTANT: Gemini 2.5 models do "thinking" by default, which
            # consumes output tokens internally before any visible text is
            # produced. With max_output_tokens=8 the thinking budget eats
            # everything and response.text comes back as None. We disable
            # thinking entirely for the judge call — it's a single-digit
            # rating, there is nothing to reason about.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    return _parse_score(response.text or "")


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
# Smoke test — uses 1 API call (free):
#     python -m src.eval.llm_judge
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    score = judge_pair(
        question="How do you list all containers, including stopped ones?",
        reference="docker ps -a. The -a flag includes stopped containers.",
        candidate="Use 'docker ps --all' to list every container including stopped ones.",
    )
    print(f"Gemini judge score (should be 4 or 5): {score}")
