"""
CLI: build data/train/train_1k.jsonl from Stack Overflow [docker] Q&A.

USAGE (from project root, with venv activated):
    python scripts/03_build_train.py
    python scripts/03_build_train.py --target 1000 --overfetch 1500
    python scripts/03_build_train.py --threshold 0.75 --tag docker

WHAT IT DOES:
    1. Fetches ~1500 top-voted SO Docker Q&A via the Stack Exchange REST API.
    2. Strips HTML, drops low-quality rows (no accepted answer, score<5, etc.).
    3. Semantically dedupes against data/eval/eval.jsonl using all-MiniLM-L6-v2
       embeddings + cosine ≥ 0.75 — enforces the disjoint rule against
       paraphrases of eval questions, not just URL collisions.
    4. Samples down to exactly --target rows (default 1000), keeping the
       highest-SO-vote ones.
    5. Writes data/train/train_1k.jsonl.
    6. Prints 30 random spot-check samples + 10 examples of what was
       semantically dropped (so you can sanity-check the threshold).

WHY OVER-FETCH:
    Stages 2 and 3 drop rows. Asking the API for ~1500 raw to land 1000 clean
    leaves headroom for typical drop rates (~15-25% combined).

NETWORK:
    The Stack Exchange API allows 300 anonymous req/IP/day; this script makes
    ~30. If you re-run more than ~10 times in a day you may need to wait or
    register an app key (see https://api.stackexchange.com/docs/authentication).
"""

import argparse
import random
import sys
from pathlib import Path

# Make `src.*` importable when running this script from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 stdout — Windows console defaults to cp1252 which mangles ≥/✓.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config
from src.data.build_eval import load_eval
from src.data.build_train import (
    fetch_so_questions,
    fetch_so_answers,
    filter_candidates,
    dedup_against_eval,
    write_train_jsonl,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target",     type=int,   default=1000,
                   help="Final number of training rows (default 1000)")
    p.add_argument("--overfetch",  type=int,   default=1500,
                   help="Raw rows to fetch from SO before filtering (default 1500)")
    p.add_argument("--threshold",  type=float, default=0.75,
                   help="Cosine-sim threshold for eval-paraphrase dedup (default 0.75)")
    p.add_argument("--tag",        type=str,   default="docker",
                   help="SO tag to query (default 'docker')")
    p.add_argument("--seed",       type=int,   default=config.SEED,
                   help="Random seed for the spot-check sampling")
    args = p.parse_args()

    random.seed(args.seed)

    # ------------------------------------------------------------------ STAGE 1
    print("=" * 70)
    print("STAGE 1/5  Fetch raw Stack Overflow Q&A")
    print("=" * 70)
    raw_questions = fetch_so_questions(target_n=args.overfetch, tag=args.tag)

    # Collect accepted_answer_ids and fetch those bodies.
    answer_ids = [q["accepted_answer_id"] for q in raw_questions
                  if q.get("accepted_answer_id")]
    print(f"  {len(answer_ids)} questions have accepted answers — fetching bodies")
    answers = fetch_so_answers(answer_ids)

    # ------------------------------------------------------------------ STAGE 2
    print("\n" + "=" * 70)
    print("STAGE 2/5  Filter for quality (score, length, HTML cleanup)")
    print("=" * 70)
    candidates = filter_candidates(raw_questions, answers)
    if len(candidates) < args.target:
        print(f"  [!] Only {len(candidates)} candidates passed quality filters "
              f"(target {args.target}). Consider raising --overfetch.")

    # ------------------------------------------------------------------ STAGE 3
    print("\n" + "=" * 70)
    print("STAGE 3/5  Semantic dedup against eval.jsonl")
    print("=" * 70)
    eval_examples = load_eval(config.EVAL_FILE)
    eval_questions = [e.question for e in eval_examples]
    print(f"  loaded {len(eval_questions)} eval questions from {config.EVAL_FILE}")

    kept, dropped = dedup_against_eval(
        candidates,
        eval_questions,
        threshold=args.threshold,
    )

    if len(kept) < args.target:
        print(f"  [!] Only {len(kept)} candidates survived dedup "
              f"(target {args.target}).")
        print(f"      Try a higher --overfetch, or a slightly higher --threshold "
              f"if you're confident the drops are false-positives.")
        # Continue anyway — we still write what we have.

    # ------------------------------------------------------------------ STAGE 4
    print("\n" + "=" * 70)
    print(f"STAGE 4/5  Sample down to {args.target} rows (top-voted)")
    print("=" * 70)
    # Sort by SO question score descending, take the top N. Highest-voted
    # questions tend to be the most general ones — better domain coverage.
    kept_sorted = sorted(kept, key=lambda r: r.score, reverse=True)
    final = kept_sorted[:args.target]
    print(f"  selected {len(final)} rows "
          f"(score range: {final[-1].score} → {final[0].score})")

    # ------------------------------------------------------------------ STAGE 5
    print("\n" + "=" * 70)
    print(f"STAGE 5/5  Write {config.TRAIN_FILE}")
    print("=" * 70)
    n_written = write_train_jsonl(final, config.TRAIN_FILE)
    print(f"  wrote {n_written} rows to {config.TRAIN_FILE}")

    # ------------------------------------------------------------------ SPOT CHECK
    print("\n" + "=" * 70)
    print("SPOT CHECK  30 random samples (read these!)")
    print("=" * 70)
    sample = random.sample(final, min(30, len(final)))
    for i, row in enumerate(sample, 1):
        print(f"\n--- sample {i}/30  (so_id={row.so_question_id}, score={row.score})")
        print(f"  Q: {row.question}")
        ans_preview = row.answer.replace("\n", " ")
        if len(ans_preview) > 200:
            ans_preview = ans_preview[:200] + "..."
        print(f"  A: {ans_preview}")

    # Show some dropped rows so you can verify the dedup threshold is sane.
    if dropped:
        print("\n" + "=" * 70)
        print("DEDUP REVIEW  10 examples of what was dropped as eval-paraphrase")
        print("=" * 70)
        sample_dropped = random.sample(dropped, min(10, len(dropped)))
        for i, (row, sim, nearest_eval) in enumerate(sample_dropped, 1):
            print(f"\n--- dropped {i}/10  (cos={sim:.3f})")
            print(f"  train Q: {row.question}")
            print(f"  eval  Q: {nearest_eval}")

    print("\n[OK] train_1k.jsonl built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
