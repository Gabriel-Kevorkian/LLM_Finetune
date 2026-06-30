"""
CLI: evaluate a FINE-TUNED LoRA adapter (base Mistral-7B + adapter) over eval.jsonl.

This is the Sunday counterpart to scripts/02_baseline_eval.py. The ONLY thing
that differs between the two is how the model is loaded:

    02 (baseline):   plain Mistral-7B-v0.3, 4-bit, no adapter.
    05 (this file):  Unsloth base + a trained LoRA adapter layered on top.

Everything downstream — prompt formatting, ROUGE/EM scoring, the Gemini
LLM-judge, the per_example.csv + results.json output — goes through the SAME
src/eval/runner.run_eval() as the baseline. That shared path is what makes the
before/after comparison fair: identical prompts, identical scoring, identical
output schema. (See the runner's module docstring.)

USAGE PATHS:

  --- Local pipeline test (FAST, no GPU, no API cost): ---
    python scripts/05_eval_finetuned.py --mock --skip-judge --run-name r16

    Uses a dummy generator that ignores the adapter and returns canned text.
    Verifies the eval plumbing (load → score → write CSV/JSON) without
    touching a GPU. Run this on your 6 GB Windows box before going to Colab.

  --- Real eval on Colab T4: ---
    python scripts/05_eval_finetuned.py --adapter models/adapters/r16 --run-name r16

    Loads the Unsloth 4-bit base, layers the r=16 LoRA adapter on top,
    generates an answer for each of the 50 eval questions, scores with
    ROUGE/EM/Gemini-judge, and writes results/runs/r16/.

    Required env vars (auto-loaded from .env):
      HF_TOKEN          - to download the base weights (Mistral is gated)
      GEMINI_API_KEY    - for the LLM-judge (gemini-3.1-flash-lite, free tier)

WHY THE BASE IS unsloth/mistral-7b-v0.3-bnb-4bit (not mistralai/...):
    The adapter was TRAINED on top of Unsloth's pre-quantized base. A LoRA
    adapter is a delta against specific base weights — to apply it correctly we
    must load the exact same base. These are the same underlying Mistral-7B-v0.3
    weights as the baseline used; only the quantization packaging differs, so
    the baseline-vs-finetuned comparison stays fair.
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
    """Returns a dummy generator. Used to verify the pipeline without a GPU."""
    def _mock(question: str) -> str:
        return (
            f"[MOCK ANSWER] I would normally answer this Docker question with "
            f"the FINE-TUNED model: '{question[:80]}'. This text exists only to "
            f"exercise the scoring pipeline."
        )
    return _mock


def make_finetuned_generate_fn(adapter_path: str, base_model_name: str):
    """Returns a generator backed by the base model + a trained LoRA adapter.

    The import of `unsloth` happens FIRST, inside this function, on purpose:
      - Importing it inside (not at module top) keeps this file usable in
        --mock mode on the local 6 GB box, where unsloth/CUDA aren't present.
      - Importing unsloth BEFORE trl/transformers/peft is mandatory — Unsloth
        monkey-patches those libraries at import time, and if they're loaded
        first the patches land on stale references and you get version-mismatch
        crashes. (See the unsloth-install-pitfalls notes.)
    """
    import unsloth  # noqa: F401  -- MUST come before transformers/peft/trl
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from src.data.format_prompts import format_chat, ensure_chat_template

    print(f"Loading base model (Unsloth 4-bit): {base_model_name}")

    # Step 1: load the SAME 4-bit base the adapter was trained from.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = base_model_name,
        max_seq_length = config.MAX_SEQ_LENGTH,
        dtype          = None,          # let Unsloth pick (bf16/fp16 per GPU)
        load_in_4bit   = config.LOAD_IN_4BIT,
    )
    ensure_chat_template(tokenizer)

    print(f"Layering LoRA adapter on top: {adapter_path}")

    # Step 2: layer the trained LoRA delta ON TOP. The adapter folder holds
    # only adapter_config.json + adapter_model.safetensors (~160 MB), NOT a
    # full standalone model — so this is PeftModel.from_pretrained, never a
    # single FastLanguageModel.from_pretrained(adapter_path) call (which would
    # fail with "No config file found").
    model = PeftModel.from_pretrained(model, adapter_path)

    # Step 3: switch into Unsloth's optimized inference mode.
    FastLanguageModel.for_inference(model)

    def _generate(question: str) -> str:
        # Identical generation settings to the baseline (scripts/02) so the
        # ONLY difference between the two runs is the adapter itself.
        prompt = format_chat(question, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        import torch
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                do_sample=False,          # greedy = deterministic
                temperature=config.EVAL_TEMPERATURE,
                # Break the runaway-repetition loops from the failure analysis.
                # Same constants used by the baseline (scripts/02) so the runs
                # stay comparable. See config.EVAL_REPETITION_PENALTY notes.
                repetition_penalty=config.EVAL_REPETITION_PENALTY,
                no_repeat_ngram_size=config.EVAL_NO_REPEAT_NGRAM_SIZE,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Slice off the prompt tokens — we only want the model's new output.
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return _generate


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to the trained LoRA adapter folder (e.g. "
             "models/adapters/r16). Required unless --mock is set.",
    )
    parser.add_argument(
        "--run-name", default="r16",
        help="Label for this run. Names the output folder "
             "(results/runs/<run-name>/) and is stored in results.json. "
             "Default: r16",
    )
    parser.add_argument(
        "--base-model", default=config.UNSLOTH_MODEL_NAME,
        help=f"HF model id for the 4-bit base the adapter was trained on. "
             f"Default: {config.UNSLOTH_MODEL_NAME}",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use a dummy generator (no GPU, no adapter load). For pipeline "
             "verification on a CPU-only / low-VRAM machine.",
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip the Gemini LLM-judge step. Useful with --mock to avoid "
             "burning judge calls on dummy answers.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory. Default: results/runs/<run-name>/",
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
        model_id = f"mock-finetuned:{args.run_name}"
    else:
        if not args.adapter:
            print("ERROR: --adapter is required for a real eval run "
                  "(or pass --mock for a pipeline test).")
            return 1
        if not Path(args.adapter).exists():
            print(f"ERROR: adapter path not found: {args.adapter}")
            print("On Colab, make sure Drive is mounted and models/adapters/ "
                  "is symlinked to your Drive folder first.")
            return 1
        generate_fn = make_finetuned_generate_fn(args.adapter, args.base_model)
        model_id = f"{args.base_model}+adapter:{args.adapter}"

    # 3. Resolve output dir. Default mirrors the folder layout in README §4:
    #    results/runs/<run-name>/  (so the Task #7 compile step can iterate).
    out_dir = Path(args.out) if args.out else (config.RUNS_DIR / args.run_name)

    # 4. Run the eval (same runner as the baseline → identical output schema).
    run_eval(
        examples=examples,
        generate_fn=generate_fn,
        output_dir=out_dir,
        run_name=args.run_name if not args.mock else f"{args.run_name}-mock",
        use_llm_judge=not args.skip_judge,
        model_id=model_id,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
