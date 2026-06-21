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

WHY GEMINI 3.1 FLASH-LITE:
    - Free tier: 15 RPM AND 500 RPD — generous enough to cover all 200
      judge calls for the weekend (baseline + 3 ablations × 50 questions)
      with a comfortable margin.
    - Compare to gemini-2.5-flash-lite (10 RPM, only 20 RPD) which would
      take 10 days to finish all our judging on free tier.
    - "Lite" variant skips internal "thinking" tokens, so the response
      arrives quickly as a clean single digit.
    - Quality is more than enough for a 1-5 rubric grading task.

WHY TEMPERATURE = 0:
    Judging must be reproducible. Same inputs -> same score every run.
    If the judge was random, our before/after comparison would be polluted
    by judge noise rather than actual model improvement.

RUBRIC DESIGN:
    A vague prompt ("rate from 1 to 5") gives noisy ratings. We define
    each score level explicitly so the judge clusters around the right
    bucket consistently.

RETRY POLICY:
    On 429 / 503 we sleep (using the server's suggested retryDelay if
    present, otherwise exponential backoff) and try again up to MAX_RETRIES
    times. Other errors propagate immediately.
"""

from __future__ import annotations

import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

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


MAX_RETRIES         = 4
DEFAULT_RETRY_FLOOR = 4   # seconds — minimum wait if server gives no delay


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


def _extract_retry_seconds(error_msg: str) -> int | None:
    """Pull the server-suggested retry delay out of a Gemini 429 error.

    The error JSON contains 'retryDelay': '11s' — we want the 11.
    Returns None if not present (caller uses its own backoff).
    """
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", error_msg)
    if match:
        # Add 1s buffer to be safe against clock skew / off-by-one quota windows.
        return int(match.group(1)) + 1
    return None


def _is_retriable(error_msg: str) -> bool:
    """True for transient errors worth retrying (rate limit, 503)."""
    return (
        "429" in error_msg
        or "RESOURCE_EXHAUSTED" in error_msg
        or "503" in error_msg
        or "UNAVAILABLE" in error_msg
    )


def judge_pair(
    question: str,
    reference: str,
    candidate: str,
    *,
    model: str = "gemini-3.1-flash-lite",
    client: genai.Client | None = None,
) -> int:
    """Return an integer score from 1 to 5.

    Falls back to 1 (worst score) if parsing fails or all retries exhausted —
    better to under-credit than to silently insert wrong scores into results.
    """
    # Guard against empty candidate strings (model refused, generation hit
    # max tokens with nothing useful, etc.). Real failure → score 1, no API call.
    if not candidate.strip():
        return 1

    c = client or _client()

    prompt = _JUDGE_PROMPT.format(
        question=question, reference=reference, candidate=candidate,
    )

    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8,
        # Disable thinking — harmless on flash-lite (which doesn't think
        # by default) but defensive in case we ever swap back to 2.5-flash.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = c.models.generate_content(
                model=model, contents=prompt, config=cfg,
            )
            return _parse_score(response.text or "")
        except Exception as e:
            last_err = e
            msg = str(e)
            if not _is_retriable(msg) or attempt >= MAX_RETRIES:
                # Non-retriable or out of attempts → propagate; the runner
                # will catch it and record None as the judge score.
                raise
            # Use the server-suggested delay if present; otherwise exponential
            # backoff starting at DEFAULT_RETRY_FLOOR.
            wait = _extract_retry_seconds(msg) or DEFAULT_RETRY_FLOOR * (2 ** attempt)
            time.sleep(wait)

    # Unreachable — defensive fallback.
    raise last_err or RuntimeError("judge_pair: unreachable code path")


def _parse_score(raw: str) -> int:
    """Extract an integer 1-5 from the model's reply.

    Robust to leading/trailing whitespace, periods, and the occasional
    'Score: 4' formatting that the model sometimes adds despite our prompt.
    """
    match = re.search(r"\b([1-5])\b", raw)
    if match is None:
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
