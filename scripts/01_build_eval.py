"""
CLI: validate data/eval/eval.jsonl and print summary stats.

USAGE (from project root, with venv activated):
    python scripts/01_build_eval.py

WHAT IT DOES:
    1. Loads eval.jsonl.
    2. Runs the validator — exits with code 1 if any errors found.
    3. Prints a small report: total count, category distribution,
       difficulty distribution, and any warnings.

WHEN TO RUN IT:
    Every time you edit eval.jsonl. The validator catches the kinds of
    mistakes (placeholder answers, missing sources, duplicate IDs) that
    would otherwise only show up DURING the baseline eval — by which point
    you've wasted Colab time.
"""

# Make `src.*` importable when running this script from the project root.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 stdout so non-ASCII characters (≥, ✓, etc.) don't crash on
# Windows' default cp1252 console. Safe no-op on Linux/Mac.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config
from src.data.build_eval import load_eval, validate_eval, summarize_eval


def main() -> int:
    eval_file = config.EVAL_FILE

    if not eval_file.exists():
        print(f"ERROR: eval file not found at {eval_file}")
        print("Create it first — see data/eval/eval.jsonl in this repo.")
        return 1

    examples = load_eval(eval_file)
    report = validate_eval(examples)
    stats = summarize_eval(examples)

    # --- Pretty print -------------------------------------------------------
    print(f"\nEval file:  {eval_file}")
    print(f"Total rows: {stats['total']}\n")

    print("By category:")
    for cat, n in sorted(stats["by_category"].items()):
        print(f"  {cat:18s} {n:3d}")

    print("\nBy difficulty:")
    for diff in ("easy", "medium", "hard"):
        n = stats["by_difficulty"].get(diff, 0)
        print(f"  {diff:18s} {n:3d}")

    if report.warnings:
        print("\nWARNINGS:")
        for w in report.warnings:
            print(f"  [!] {w}")

    if report.errors:
        print(f"\nERRORS ({len(report.errors)} total — must fix before training):")
        for e in report.errors:
            print(f"  [X] {e}")
        return 1

    print("\n[OK] eval.jsonl is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
