"""
Evaluation orchestration: run a model over the eval set and save results.

WHY THIS MODULE EXISTS:
    Every evaluation in the project (baseline, fine-tuned r=16, ablation
    r=8, r=32, ...) goes through this SAME function. The only thing that
    changes between runs is the `generate_fn` we pass in. This guarantees:

      1. The same prompts get fed in the same way.
      2. The same scoring code is used.
      3. Per-example CSV and summary JSON have identical schemas across
         all runs, which means the Sunday "compile results" step can just
         iterate over folders without special-casing each model.

OUTPUT FILES:
    For a given output_dir (e.g. results/baseline/), we write:

      - per_example.csv : one row per eval question with full text +
                          all scores. Used for failure analysis.
      - results.json    : summary aggregates (overall + per category +
                          per difficulty). Used for the comparison table.

    These two are the universal output format. Every run produces them.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from src.data.build_eval import EvalExample
from src.eval.metrics import score_pair
from src.eval.llm_judge import judge_pair


# Type alias: a generate_fn takes a single prompt string and returns the
# model's completion as a string. We deliberately keep it simple — batching
# can be added later if the eval becomes a bottleneck (it won't at N=50).
GenerateFn = Callable[[str], str]


def run_eval(
    examples: list[EvalExample],
    generate_fn: GenerateFn,
    output_dir: Path,
    run_name: str,
    *,
    use_llm_judge: bool = True,
    judge_model: str = "gemini-3.1-flash-lite",
    model_id: str = "unknown",
) -> dict:
    """Run `generate_fn` over every eval example, score the outputs, save results.

    Args:
        examples       : list of EvalExample (typically the full eval.jsonl).
        generate_fn    : called with a formatted prompt, returns the model's
                         answer text. The CALLER decides what model — base,
                         fine-tuned, mock, whatever.
        output_dir     : where to write per_example.csv and results.json.
        run_name       : label for this run, stored in results.json so we
                         can identify it later (e.g. "baseline", "r16").
        use_llm_judge  : if False, skip the GPT-4o-mini scoring step (saves
                         API calls during pipeline development).
        judge_model    : Gemini model id for the judge (default Flash).
        model_id       : free-text identifier for the model under test.

    Returns:
        The summary dict that was also written to results.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = output_dir / "per_example.csv"
    summary_path     = output_dir / "results.json"

    rows: list[dict] = []

    print(f"\n=== Running eval: {run_name} ===")
    print(f"Model id:      {model_id}")
    print(f"Examples:      {len(examples)}")
    print(f"LLM judge:     {'on' if use_llm_judge else 'off'}")
    print(f"Output dir:    {output_dir}")

    for ex in tqdm(examples, desc=run_name, ncols=80):
        # Step 1: ask the model for its answer.
        # NOTE: generate_fn is expected to handle prompt formatting (chat
        # template) internally — we just hand it the raw question. The
        # Colab notebook will wrap a real tokenizer around it.
        try:
            prediction = generate_fn(ex.question)
        except Exception as e:
            # Don't let one bad generation kill the whole run.
            prediction = f"[GENERATION_ERROR: {type(e).__name__}: {e}]"

        # Step 2: compute ROUGE + exact match (cheap, local).
        scores = score_pair(prediction, ex.reference_answer)

        # Step 3: optionally call the LLM judge (slower, costs $).
        judge_score: int | None = None
        if use_llm_judge:
            try:
                judge_score = judge_pair(
                    question=ex.question,
                    reference=ex.reference_answer,
                    candidate=prediction,
                    model=judge_model,
                )
            except Exception as e:
                # Network blip on one example shouldn't tank the whole run.
                print(f"\n[judge error on {ex.id}: {e}]")
                judge_score = None

        rows.append({
            "id":               ex.id,
            "category":         ex.category,
            "difficulty":       ex.difficulty,
            "question":         ex.question,
            "reference":        ex.reference_answer,
            "prediction":       prediction,
            "rouge1":           round(scores["rouge1"], 4),
            "rougeL":           round(scores["rougeL"], 4),
            "exact_match":      scores["exact_match"],
            "judge_score":      judge_score,
        })

    # --- write per_example.csv ---------------------------------------------
    with open(per_example_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- compute summary ---------------------------------------------------
    summary = _summarize(rows, run_name=run_name, model_id=model_id,
                         use_llm_judge=use_llm_judge)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- pretty-print headline numbers -------------------------------------
    print("\n--- Summary ---")
    agg = summary["aggregate"]
    print(f"  rouge1:      {agg['rouge1']:.4f}")
    print(f"  rougeL:      {agg['rougeL']:.4f}")
    print(f"  exact_match: {agg['exact_match']:.4f}")
    if use_llm_judge:
        print(f"  judge_score: {agg['judge_score']:.2f} / 5")
    print(f"\nWrote: {per_example_path}")
    print(f"Wrote: {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _summarize(rows: list[dict], *, run_name: str, model_id: str,
               use_llm_judge: bool) -> dict:
    """Aggregate per-example scores into a JSON-serializable summary."""

    def _mean(key: str, subset: list[dict]) -> float | None:
        values = [r[key] for r in subset if r[key] is not None]
        return mean(values) if values else None

    def _agg(subset: list[dict]) -> dict:
        return {
            "n":           len(subset),
            "rouge1":      _mean("rouge1", subset),
            "rougeL":      _mean("rougeL", subset),
            "exact_match": _mean("exact_match", subset),
            "judge_score": _mean("judge_score", subset) if use_llm_judge else None,
        }

    by_category   = {c: _agg([r for r in rows if r["category"] == c])
                     for c in sorted({r["category"] for r in rows})}
    by_difficulty = {d: _agg([r for r in rows if r["difficulty"] == d])
                     for d in ("easy", "medium", "hard")
                     if any(r["difficulty"] == d for r in rows)}

    return {
        "run_name":       run_name,
        "model_id":       model_id,
        "num_examples":   len(rows),
        "use_llm_judge":  use_llm_judge,
        "aggregate":      _agg(rows),
        "by_category":    by_category,
        "by_difficulty":  by_difficulty,
    }
