"""
Run a LoRA fine-tune on data/train/train_1k.jsonl and save the adapter.

WHAT THIS FILE DOES (end to end):
    1. Read the YAML config (rank, lr, epochs, etc.) and the training JSONL.
    2. Load Mistral-7B in 4-bit + attach LoRA adapters (via src.training.
       load_model.load_model_for_training).
    3. Turn each JSONL row into a chat-formatted string via format_chat_pair
       (uses the same helper as eval — guarantees train/eval format match).
    4. Wrap the formatted strings as a HuggingFace Dataset.
    5. Run TRL's SFTTrainer for `num_epochs` epochs.
    6. Save the LoRA adapter to models/adapters/<run_name>/.
    7. Copy the YAML and write a training summary into
       results/runs/<run_name>/ so the run is reproducible.

WHY USE TRL's SFTTrainer:
    HuggingFace's standard Trainer is for general supervised learning. TRL's
    SFTTrainer is a thin wrapper around it that handles the
    "supervised fine-tuning on text completion pairs" use case specifically:
    it knows how to tokenize a chat-formatted string, set up labels for
    next-token prediction, and play nicely with PEFT-wrapped (LoRA) models.

    Could we write this loop by hand? Yes, in ~40 lines of PyTorch. But:
    - SFTTrainer integrates with W&B logging out of the box.
    - It handles gradient accumulation, mixed precision, lr scheduling.
    - One less file to maintain.

LOSS MASKING — A KNOWN SIMPLIFICATION:
    By default SFTTrainer computes loss on the WHOLE chat string, including
    the user question. The pedantically-correct setup is to mask the question
    tokens and only compute loss on the assistant's answer (using
    DataCollatorForCompletionOnlyLM). For a weekend project with structured
    Docker Q&A, this distinction is small — the model still learns to produce
    answers in the right style — and the extra collator code adds complexity.
    Documented here so the failure_analysis.md can mention it if relevant.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# -----------------------------------------------------------------------------
# Config dataclass — what comes out of configs/base.yaml
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Frozen view of a training YAML. One source of truth for hyperparameters."""
    run_name:               str
    lora_rank:              int
    lora_alpha:             int
    lora_dropout:           float
    lora_target_modules:    list[str]
    learning_rate:          float
    num_epochs:             int
    per_device_batch_size:  int
    grad_accum_steps:       int
    warmup_steps:           int
    seed:                   int
    max_seq_length:         int
    logging_steps:          int
    save_steps:             int
    wandb_enabled:          bool


def load_train_config(path: Path) -> TrainConfig:
    """Read a configs/*.yaml file into a TrainConfig dataclass.

    Doing this as a dataclass (instead of a raw dict) gives us:
      - one place that lists every expected field (the dataclass itself)
      - a TypeError immediately if the YAML is missing a required key,
        instead of a confusing KeyError 200 lines later inside the trainer.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainConfig(**raw)


# -----------------------------------------------------------------------------
# Dataset construction — JSONL rows → chat-templated text the trainer expects
# -----------------------------------------------------------------------------
def _load_train_jsonl(path: Path) -> list[dict]:
    """Load train_1k.jsonl into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _format_rows_to_chat(rows: list[dict], tokenizer: Any) -> list[dict]:
    """Convert each {question, answer, ...} row into {"text": <chat string>}.

    Why a `text` field specifically: SFTTrainer accepts a `dataset_text_field`
    parameter and tokenizes that column. By pre-formatting here, the trainer
    just has to tokenize — no chat-template logic inside the training loop.

    This is the SAME helper the eval script uses (format_chat_pair) — that's
    deliberate. If train and eval use different chat formats, the model gets
    artificially low eval scores because it's seeing a format it never trained
    on. Going through one helper guarantees consistency.
    """
    from src.data.format_prompts import format_chat_pair

    out = []
    for r in rows:
        chat_text = format_chat_pair(
            question = r["question"],
            answer   = r["answer"],
            tokenizer= tokenizer,
        )
        out.append({"text": chat_text})
    return out


# -----------------------------------------------------------------------------
# Main entry — called by scripts/04_train.py
# -----------------------------------------------------------------------------
def run_training(
    config:        TrainConfig,
    train_file:    Path,
    base_model_id: str,
    adapters_dir:  Path,
    results_dir:   Path,
    yaml_path:     Path,
) -> dict:
    """Execute one fine-tuning run end-to-end.

    Args:
        config:        Parsed TrainConfig (from load_train_config).
        train_file:    Path to data/train/train_1k.jsonl.
        base_model_id: HF model ID to load (e.g. unsloth/mistral-7b-v0.3-bnb-4bit).
        adapters_dir:  Where models/adapters/ lives. Adapter saves to
                       {adapters_dir}/{config.run_name}/.
        results_dir:   Where results/runs/ lives. YAML + summary land in
                       {results_dir}/{config.run_name}/.
        yaml_path:     Path to the source YAML, copied into results/runs/...
                       for reproducibility.

    Returns:
        A dict with the final loss, step count, and output paths.
    """
    # Lazy imports — torch/datasets/trl can't be installed on the user's
    # local Windows box (CUDA), only on Colab. This function is meant to run
    # ON COLAB; importing the module locally must not crash.
    import torch
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from src.training.load_model import load_model_for_training

    # --- 1. Load model + tokenizer (loads quant base + attaches LoRA) -------
    model, tokenizer = load_model_for_training(
        base_model_name     = base_model_id,
        max_seq_length      = config.max_seq_length,
        lora_rank           = config.lora_rank,
        lora_alpha          = config.lora_alpha,
        lora_dropout        = config.lora_dropout,
        lora_target_modules = config.lora_target_modules,
        seed                = config.seed,
    )

    # --- 2. Load training data and turn into chat-formatted Dataset --------
    rows = _load_train_jsonl(train_file)
    print(f"Loaded {len(rows)} training examples from {train_file.name}")

    formatted = _format_rows_to_chat(rows, tokenizer)
    dataset   = Dataset.from_list(formatted)

    # Show one fully-formatted example so we can eyeball the chat template
    # is correct before we spend an hour training on it.
    print("\n--- example formatted row (first 800 chars) ---")
    print(formatted[0]["text"][:800])
    print("--- end example ---\n")

    # --- 3. Set up training arguments --------------------------------------
    # `bf16` (bfloat16): T4 supports it via torch's autocast wrapping. bf16
    # has the same exponent range as fp32 (so it doesn't overflow on large
    # logits the way fp16 does) but half the bits. It's the standard choice
    # for LoRA training on modern GPUs.
    #
    # `optim="adamw_8bit"`: AdamW maintains TWO extra tensors per param (the
    # running mean + variance of gradients). At full precision that doubles
    # the VRAM cost of the model. bitsandbytes' 8-bit AdamW stores those
    # tensors quantized — saves ~3 GB VRAM. Crucial fit on T4.
    #
    # `report_to`: "wandb" only if the user opted in. Otherwise "none" to
    # avoid the trainer trying to log to a service we didn't configure.
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not bf16_ok:
        print("[WARN] bf16 not supported — falling back to fp16 (may be less stable)")

    training_args = TrainingArguments(
        output_dir                  = str(adapters_dir / config.run_name / "_checkpoints"),
        per_device_train_batch_size = config.per_device_batch_size,
        gradient_accumulation_steps = config.grad_accum_steps,
        num_train_epochs            = config.num_epochs,
        learning_rate               = config.learning_rate,
        warmup_steps                = config.warmup_steps,
        logging_steps               = config.logging_steps,
        save_strategy               = "no" if config.save_steps == 0 else "steps",
        save_steps                  = config.save_steps if config.save_steps > 0 else 500,
        seed                        = config.seed,
        bf16                        = bf16_ok,
        fp16                        = not bf16_ok,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        report_to                   = ["wandb"] if config.wandb_enabled else ["none"],
        run_name                    = config.run_name,
    )

    # --- 4. Build the SFTTrainer -------------------------------------------
    # `dataset_text_field="text"`: tells SFTTrainer which column of our
    #     Dataset holds the formatted string to tokenize.
    # `packing=False`: don't concatenate multiple examples into one sequence.
    #     Packing speeds training up at the cost of cross-example contamination
    #     on the loss. With only 1K rows and a weekend deadline, the simpler
    #     unpacked setup is the right tradeoff.
    # `max_seq_length`: examples longer than this get truncated. 2048 fits
    #     well within T4's VRAM and covers >99% of Docker Q&A.
    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        dataset_text_field = "text",
        max_seq_length     = config.max_seq_length,
        dataset_num_proc   = 2,
        packing            = False,
        args               = training_args,
    )

    # --- 5. Train ----------------------------------------------------------
    print("Starting training...")
    train_result = trainer.train()
    final_loss = float(train_result.training_loss)
    print(f"Training done — final loss = {final_loss:.4f}")

    # --- 6. Save the LoRA adapter ------------------------------------------
    # Note: this saves ONLY the LoRA matrices A and B (+ tokenizer config),
    # not the full 7B base model. Resulting folder is ~80 MB for r=16.
    # To use it at eval time: load base Mistral + PeftModel.from_pretrained
    # pointing at this folder.
    adapter_out = adapters_dir / config.run_name
    adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    print(f"Adapter saved to {adapter_out}")

    # --- 7. Freeze the config + summary in results/runs/<run_name>/ --------
    # This is the reproducibility audit trail. Anyone reviewing the project
    # can open results/runs/r16/ and see the EXACT YAML that produced these
    # numbers, plus a one-line summary.
    run_out = results_dir / config.run_name
    run_out.mkdir(parents=True, exist_ok=True)
    shutil.copy(yaml_path, run_out / "config.yaml")

    summary = {
        "run_name":      config.run_name,
        "final_loss":    final_loss,
        "global_step":   int(train_result.global_step),
        "num_examples":  len(rows),
        "adapter_path":  str(adapter_out),
        "base_model":    base_model_id,
    }
    (run_out / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Run artifacts saved to {run_out}")

    return summary
