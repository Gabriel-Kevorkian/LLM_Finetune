"""
CLI: compile all eval runs into one ablation table + a rank-vs-quality chart.

This is README §3 Sun PM-1. After the baseline and the three fine-tuned runs
(r8/r16/r32) have each been scored, every run has written a results.json with
an identical schema (see src/eval/runner.py). This script gathers them into:

  - results/ablation_table.csv   : one row per model (baseline + each rank),
                                    columns rank / rouge1 / rougeL / EM / judge.
  - results/charts/rank_vs_rouge.png : LoRA rank (x) vs ROUGE-L & ROUGE-1 (left
                                    y) and judge score (right y), with the
                                    rank-independent baseline drawn as dashed
                                    reference lines.

WHERE EACH RUN'S RANK COMES FROM:
    results.json stores scores but NOT the LoRA rank. The rank lives in the
    frozen config.yaml the training script copied into results/runs/<name>/.
    We read lora_rank from there (and fall back to parsing the digits out of
    the run_name, e.g. "r8" -> 8, if no config.yaml is present — which is the
    case for mock runs).

ROBUST TO MISSING RUNS:
    You can run this any time. If only baseline + r16 exist so far, you get a
    2-row table and a chart with a single rank point. Re-run after r8/r32 land
    to fill it in. Folders without a results.json are skipped with a note.

USAGE:
    python scripts/06_compile_results.py
    python scripts/06_compile_results.py --no-chart        # CSV only
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Make `src.*` importable when running from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config

# Columns in the order we want them in the CSV.
_FIELDS = ["run_name", "rank", "n", "rouge1", "rougeL", "exact_match", "judge_score"]


def _read_results_json(path: Path) -> dict | None:
    """Load a run's results.json, returning None if it's missing/unreadable."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [skip] {path}: {e}")
        return None


def _rank_for_run(run_dir: Path, run_name: str) -> int | None:
    """Resolve the LoRA rank for a fine-tuned run.

    Prefer the frozen config.yaml (authoritative). Fall back to the integer
    embedded in the run_name (e.g. "r32" -> 32). Returns None if neither yields
    a rank (shouldn't happen for real runs).
    """
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg and "lora_rank" in cfg:
                return int(cfg["lora_rank"])
        except Exception as e:  # noqa: BLE001 - config is best-effort
            print(f"  [warn] couldn't read rank from {cfg_path}: {e}")
    m = re.search(r"\d+", run_name)
    return int(m.group()) if m else None


def _row_from_summary(summary: dict, *, rank: int | None) -> dict:
    """Flatten a results.json summary into one CSV row."""
    agg = summary["aggregate"]
    return {
        "run_name":    summary.get("run_name", "?"),
        "rank":        rank,
        "n":           agg.get("n"),
        "rouge1":      agg.get("rouge1"),
        "rougeL":      agg.get("rougeL"),
        "exact_match": agg.get("exact_match"),
        "judge_score": agg.get("judge_score"),
    }


def collect_rows(baseline_dir: Path, runs_dir: Path) -> list[dict]:
    """Gather the baseline row + one row per fine-tuned run that has results."""
    rows: list[dict] = []

    # Baseline (rank-independent — leave rank blank so it doesn't plot as a point).
    base = _read_results_json(baseline_dir / "results.json")
    if base is not None:
        rows.append(_row_from_summary(base, rank=None))
        print(f"  [ok] baseline  -> rouge1={base['aggregate'].get('rouge1')}")
    else:
        print(f"  [missing] {baseline_dir / 'results.json'} — baseline not compiled yet")

    # Fine-tuned runs: every subfolder of results/runs/ with a results.json.
    if runs_dir.exists():
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            summary = _read_results_json(run_dir / "results.json")
            if summary is None:
                print(f"  [missing] {run_dir.name}/results.json — not evaluated yet")
                continue
            rank = _rank_for_run(run_dir, summary.get("run_name", run_dir.name))
            rows.append(_row_from_summary(summary, rank=rank))
            print(f"  [ok] {run_dir.name:8s} -> rank={rank} "
                  f"rouge1={summary['aggregate'].get('rouge1')}")

    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv} ({len(rows)} rows)")


def make_chart(rows: list[dict], out_png: Path) -> bool:
    """Plot rank vs ROUGE-L/ROUGE-1 (left) + judge (right). Returns True if drawn."""
    ft = sorted((r for r in rows if r["rank"] is not None), key=lambda r: r["rank"])
    if not ft:
        print("[chart] no fine-tuned runs with a rank yet — skipping chart.")
        return False

    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed (Windows / Colab / CI)
    import matplotlib.pyplot as plt

    ranks = [r["rank"] for r in ft]
    rougeL = [r["rougeL"] for r in ft]
    rouge1 = [r["rouge1"] for r in ft]
    judge = [r["judge_score"] for r in ft]
    baseline = next((r for r in rows if r["rank"] is None), None)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    ax1.plot(ranks, rougeL, "o-", color="#1f77b4", label="ROUGE-L")
    ax1.plot(ranks, rouge1, "s--", color="#4c9be8", label="ROUGE-1")
    ax1.set_xlabel("LoRA rank")
    ax1.set_ylabel("ROUGE F1")
    ax1.set_xticks(ranks)

    # Judge score on a second y-axis (different scale, 1–5).
    ax2 = ax1.twinx()
    if all(j is not None for j in judge):
        ax2.plot(ranks, judge, "^-", color="#d62728", label="LLM judge (1–5)")
    ax2.set_ylabel("LLM judge (1–5)")

    # Baseline reference lines (rank-independent) so the lift is visible.
    if baseline is not None:
        if baseline.get("rougeL") is not None:
            ax1.axhline(baseline["rougeL"], ls=":", color="#1f77b4", alpha=0.6)
        if baseline.get("judge_score") is not None:
            ax2.axhline(baseline["judge_score"], ls=":", color="#d62728", alpha=0.6)
        ax1.text(ranks[0], baseline["rougeL"], "  baseline ROUGE-L",
                 va="bottom", fontsize=8, color="#1f77b4")

    # Merge the two axes' legends into one box.
    lines, labels = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lab2, loc="lower right", fontsize=8)

    ax1.set_title("LoRA rank vs eval quality (Docker Q&A, 1K train)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-dir", type=Path, default=config.BASELINE_DIR)
    p.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    p.add_argument("--out-csv", type=Path, default=config.RESULTS_DIR / "ablation_table.csv")
    p.add_argument("--chart", type=Path, default=config.CHARTS_DIR / "rank_vs_rouge.png")
    p.add_argument("--no-chart", action="store_true", help="Skip the PNG chart.")
    args = p.parse_args()

    print("Compiling eval results...")
    rows = collect_rows(args.baseline_dir, args.runs_dir)
    if not rows:
        print("No results found. Run the baseline / fine-tuned evals first.")
        return 1

    write_csv(rows, args.out_csv)

    # Echo the table to the console for a quick eyeball.
    print("\n=== ablation table ===")
    hdr = f"{'run':10s} {'rank':>4s} {'n':>3s} {'rouge1':>7s} {'rougeL':>7s} {'judge':>6s}"
    print(hdr)
    for r in rows:
        rank = "" if r["rank"] is None else r["rank"]
        judge = "" if r["judge_score"] is None else f"{r['judge_score']:.2f}"
        r1 = "" if r["rouge1"] is None else f"{r['rouge1']:.4f}"
        rl = "" if r["rougeL"] is None else f"{r['rougeL']:.4f}"
        print(f"{r['run_name']:10s} {str(rank):>4s} {str(r['n']):>3s} {r1:>7s} {rl:>7s} {judge:>6s}")

    if not args.no_chart:
        make_chart(rows, args.chart)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
