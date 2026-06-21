"""
CLI: run the baseline evaluation (untrained Mistral-7B) over eval.jsonl.

USAGE PATHS:

  --- Local pipeline test (FAST, no GPU, no API cost): ---
    python scripts/02_baseline_eval.py --mock --skip-judge

    Uses a dummy generator that returns canned text. Verifies the eval
    plumbing (loading, scoring, writing CSV/JSON) without touching a
    real model. Run this on your 6 GB Windows box.

  --- Real baseline on Colab T4: ---
    python scripts/02_baseline_eval.py

    Loads Mistral-7B-v0.3 (4-bit quantized via unsloth), generates an
    answer for each of the 50 eval questions, scores with ROUGE/EM/LLM-judge,
    saves to results/baseline/.

    Required env vars (load from .env automatically):
      HF_TOKEN          - to download Mistral (it's gated)
      GEMINI_API_KEY    - for the LLM-judge (Gemini 2.5 Flash, free tier)

    Expected runtime on a Colab T4:
      ~5-10 minutes (mostly model loading + 50 generations).
"""

import argparse
import sys
from pathlib import Path

# Make `src.*` importable when running from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 stdout so any unicode in print() doesn't crash on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config
from src.data.build_eval import load_eval
from src.eval.runner import run_eval


def make_mock_generate_fn():
    """Returns a dummy generator. Used to verify the pipeline without GPU."""
    def _mock(question: str) -> str:
        return (
            f"[MOCK ANSWER] I would normally answer this Docker question: "
            f"'{question[:80]}'. This text exists only to exercise the "
            f"scoring pipeline."
        )
    return _mock


def make_real_generate_fn(model_name: str):
    """Returns a generator backed by a real HuggingFace model.

    NOTE: this import happens INSIDE the function on purpose. We don't want
    `import transformers` at module load — that would fail on the local
    6 GB Windows box without bitsandbytes installed. By importing only
    when --mock is NOT set, the file stays usable in mock mode locally.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.data.format_prompts import format_chat

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name} (this takes ~1-2 minutes)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # let HF place layers on GPU
        load_in_4bit=config.LOAD_IN_4BIT,
    )
    model.eval()

    def _generate(question: str) -> str:
        prompt = format_chat(question, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                do_sample=False,          # greedy = deterministic
                temperature=config.EVAL_TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Slice off the prompt tokens — we only want the model's new output.
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return _generate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mock", action="store_true",
        help="Use a dummy generator (no GPU, no model load). For pipeline "
             "verification on a CPU-only / low-VRAM machine.",
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip the GPT-4o-mini LLM-judge scoring step. Useful with "
             "--mock to avoid burning Gemini calls on dummy answers.",
    )
    parser.add_argument(
        "--model", default=config.BASE_MODEL_NAME,
        help=f"HuggingFace model id. Default: {config.BASE_MODEL_NAME}",
    )
    parser.add_argument(
        "--out", default=str(config.BASELINE_DIR),
        help=f"Output directory. Default: {config.BASELINE_DIR}",
    )
    args = parser.parse_args()

    # 1. Load the locked eval set.
    if not config.EVAL_FILE.exists():
        print(f"ERROR: {config.EVAL_FILE} not found. Build it first (Task #2).")
        return 1

    examples = load_eval(config.EVAL_FILE)
    print(f"Loaded {len(examples)} eval examples from {config.EVAL_FILE}")

    # 2. Build the generator.
    if args.mock:
        generate_fn = make_mock_generate_fn()
        model_id = "mock"
    else:
        generate_fn = make_real_generate_fn(args.model)
        model_id = args.model

    # 3. Run the eval.
    run_eval(
        examples=examples,
        generate_fn=generate_fn,
        output_dir=Path(args.out),
        run_name="baseline" if not args.mock else "baseline-mock",
        use_llm_judge=not args.skip_judge,
        model_id=model_id,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
