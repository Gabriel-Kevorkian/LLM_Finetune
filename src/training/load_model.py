"""
Load Mistral-7B-v0.3 in 4-bit and attach LoRA adapters, ready for training.

WHY THIS IS ITS OWN FILE:
    The "load a model for training" recipe involves several knobs that all
    have to agree (4-bit quant scheme, LoRA rank/alpha/targets, sequence
    length, gradient checkpointing). Bundling them in one place means
    train_lora.py doesn't have to know any of the internals — it just calls
    `load_model_for_training(config)` and gets back a ready-to-train model.

WHY UNSLOTH (and not plain transformers + peft):
    Unsloth is a third-party library that re-implements attention + LoRA in
    custom CUDA kernels. On a T4 it gives ~2x faster training and ~30% less
    VRAM than the equivalent transformers + peft setup. It also packages
    Mistral as a pre-quantized 4-bit checkpoint ("unsloth/mistral-7b-v0.3-
    bnb-4bit") which downloads in ~1 minute vs ~5 for the full bf16 weights.

    Its API mirrors transformers' FastLanguageModel.from_pretrained, so if
    Unsloth ever breaks on a future Colab kernel you can swap to plain
    transformers + peft without changing this function's signature.

NOTE on env:
    `unsloth`, `bitsandbytes`, and `peft` are NOT in requirements.txt because
    they require CUDA and we can't install them on the user's 6 GB Windows
    box. They're installed inside the Colab notebook (see
    notebooks/colab_train.ipynb). Importing this file LOCALLY will fail with
    ModuleNotFoundError on `unsloth` — that's expected. This module is meant
    to run on Colab only.
"""

from __future__ import annotations

from typing import Any


def load_model_for_training(
    *,
    base_model_name:     str,
    max_seq_length:      int,
    lora_rank:           int,
    lora_alpha:          int,
    lora_dropout:        float,
    lora_target_modules: list[str],
    seed:                int,
) -> tuple[Any, Any]:
    """Load a Mistral-7B 4-bit base + attach LoRA adapters.

    Args:
        base_model_name:     HF model ID. For training we recommend the
                             Unsloth pre-quantized variant ("unsloth/
                             mistral-7b-v0.3-bnb-4bit") for download speed.
        max_seq_length:      Truncate examples longer than this many tokens.
                             2048 is the standard QLoRA setting on T4.
        lora_rank:           The rank `r` — see configs/base.yaml.
        lora_alpha:          LoRA scaling factor. Conventionally == lora_rank.
        lora_dropout:        Dropout on the LoRA path. 0 with Unsloth is fast
                             and stable on small datasets.
        lora_target_modules: Names of weight matrices to inject LoRA into
                             (q_proj, k_proj, etc.). See configs/base.yaml.
        seed:                Random seed for LoRA matrix init. Same seed
                             across the ablation runs = controlled comparison.

    Returns:
        (model, tokenizer) — model already has LoRA attached, tokenizer
        already has Mistral's chat template installed.
    """
    # Lazy import so the file can be byte-compiled/linted on the user's local
    # Windows box where unsloth is not available. The actual import only runs
    # when this function is called (on Colab).
    from unsloth import FastLanguageModel

    # Lazy import to keep the local CPU-only environment importable. Our
    # ensure_chat_template helper is what makes the base Mistral tokenizer
    # accept apply_chat_template() despite the base model shipping without one.
    from src.data.format_prompts import ensure_chat_template

    # --- Step 1: load the quantized base ------------------------------------
    # Under the hood, FastLanguageModel.from_pretrained:
    #   1. Downloads the .safetensors weights (already 4-bit-quantized in the
    #      Unsloth checkpoint — saves us a quantization step).
    #   2. Loads weights onto GPU with bnb's 4-bit dequantization wrapping.
    #   3. Replaces standard attention/MLP modules with Unsloth's
    #      kernel-optimized versions.
    #   4. Returns model + tokenizer.
    print(f"Loading base model: {base_model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = base_model_name,
        max_seq_length = max_seq_length,
        dtype          = None,      # None = auto-detect (bfloat16 on T4)
        load_in_4bit   = True,      # Unsloth still accepts this top-level kwarg
    )

    # --- Step 2: install Mistral chat template if missing -------------------
    # The base Mistral tokenizer ships without a chat_template. TRL's
    # SFTTrainer calls apply_chat_template under the hood, which would crash.
    # This is the same helper our eval pipeline uses — keeps train/eval
    # formatting identical.
    ensure_chat_template(tokenizer)

    # --- Step 3: attach LoRA adapters ---------------------------------------
    # get_peft_model wraps every target weight matrix with the LoRA detour:
    #     W·x  →  W·x + (alpha/rank) · (B · A) · x
    # `A` and `B` are the small trainable matrices. The frozen base `W` keeps
    # its 4-bit weights — only `A` and `B` get gradient updates.
    #
    # `bias="none"` means we do NOT train bias terms — they're tiny and rarely
    # help. Standard QLoRA setting.
    #
    # `use_gradient_checkpointing="unsloth"` enables Unsloth's memory-saving
    # mode: it discards mid-forward activations and recomputes them on the
    # backward pass. ~25-30% slower per step but ~30% less peak VRAM. Crucial
    # for fitting r=32 in 15 GB on T4.
    print(f"Attaching LoRA adapters (rank={lora_rank}, alpha={lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = lora_rank,
        target_modules             = lora_target_modules,
        lora_alpha                 = lora_alpha,
        lora_dropout               = lora_dropout,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = seed,
        use_rslora                 = False,  # plain LoRA — keeps comparison clean
        loftq_config               = None,
    )

    # Print a quick parameter summary — useful sanity check that LoRA is
    # actually attached. Expect ~40M trainable / ~7B total at r=16.
    trainable, total = _count_parameters(model)
    print(
        f"Trainable params: {trainable:,} "
        f"({100 * trainable / total:.2f}% of total {total:,})"
    )

    return model, tokenizer


def _count_parameters(model: Any) -> tuple[int, int]:
    """Return (trainable_param_count, total_param_count)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total
