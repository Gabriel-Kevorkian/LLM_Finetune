"""
Scoring functions: ROUGE and Exact Match.

WHAT IS ROUGE:
    Recall-Oriented Understudy for Gisting Evaluation. Originally designed
    for summarization, now widely used for any text generation task.
    It compares a candidate (model output) to a reference (gold answer)
    by counting overlapping n-grams.

    ROUGE-1 = overlap of unigrams (single words).
    ROUGE-2 = overlap of bigrams (word pairs).
    ROUGE-L = longest common subsequence.

    Each is reported as Precision, Recall, and F1. F1 is the
    harmonic mean — that's what we report.

    Values are between 0 and 1. Higher = more overlap. A score of 1.0
    means perfect token match (very rare for open-ended Q&A).

WHY BOTH ROUGE-1 AND ROUGE-L:
    - ROUGE-1 is lenient. It catches any word overlap, even with messed
      up order. Useful but easy to game.
    - ROUGE-L cares about ordering — it rewards keeping the same sequence
      of words as the reference. Harder to game with random word soup.
    - Reporting BOTH lets us see if the model is producing relevant words
      (R-1 high, R-L low → right vocabulary, wrong structure) or also
      gets the structure right (both high).

WHAT EXACT MATCH IS:
    1 if the prediction equals the reference (case-insensitive, whitespace
    normalized), else 0. For open-ended Q&A this is usually 0, but for
    short factual answers ("docker ps -a") it's a good sanity check.

NOTE ON STEMMING:
    use_stemmer=True means "run" and "running" count as a match. This is
    usually what you want for generative tasks; tutorials sometimes
    disable it which makes scores look worse than they really are.
"""

from __future__ import annotations

import re

from rouge_score import rouge_scorer

# Build the scorer ONCE and reuse it. Constructing a scorer parses the
# tokenizer rules and is mildly expensive — don't do it per-call.
_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace. Used for exact match only.

    We do NOT pre-normalize for ROUGE because the rouge_score library
    already applies its own tokenization and lowercasing.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalized strings match exactly, else 0.0."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """Return ROUGE-1 and ROUGE-L F1 scores between prediction and reference.

    Returns:
        {"rouge1": float in [0,1], "rougeL": float in [0,1]}
    """
    scores = _ROUGE.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def score_pair(prediction: str, reference: str) -> dict[str, float]:
    """Compute all non-LLM scores for one (prediction, reference) pair.

    Returns a dict with rouge1, rougeL, exact_match. LLM-judge is in a
    separate module because it needs network + API key.
    """
    out = rouge_scores(prediction, reference)
    out["exact_match"] = exact_match(prediction, reference)
    return out


# ---------------------------------------------------------------------------
# Smoke test — run this file directly to confirm metrics work locally.
#     python -m src.eval.metrics
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("docker ps -a", "docker ps -a"),              # exact match
        ("docker ps -a lists all containers",          # paraphrase
         "docker ps -a shows every container"),
        ("the quick brown fox",                        # zero overlap
         "completely unrelated answer"),
    ]
    for pred, ref in cases:
        s = score_pair(pred, ref)
        print(f"\nPRED: {pred!r}\nREF : {ref!r}\n-> {s}")
