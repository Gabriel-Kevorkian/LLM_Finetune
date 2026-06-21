"""
Prompt formatting helpers.

WHY THIS EXISTS:
    LLMs don't see raw text — they see specially formatted "chat templates"
    with role markers (system / user / assistant) and special tokens that
    tell the model when each turn starts and ends.

    Each model family uses DIFFERENT special tokens. Mistral uses [INST]...
    [/INST]. Llama-3 uses <|start_header_id|>user<|end_header_id|>. Qwen uses
    <|im_start|>user...<|im_end|>. If you hardcode the wrong format you'll
    get garbled output even from a perfectly-trained model.

    The correct way is to call tokenizer.apply_chat_template(), which uses
    the format the model was actually trained on. This file centralizes
    those calls.

WHY USE THE SAME FORMATTER FOR EVAL AND TRAINING:
    The model only learns the pattern it sees during training. If we train
    with one format and eval with another, the eval scores will be
    artificially low. By going through the same format_chat() helper in
    both places, we guarantee consistency.
"""

from __future__ import annotations

from typing import Any


def format_chat(question: str, tokenizer: Any) -> str:
    """Turn a single user question into the model's chat template.

    Args:
        question:  the raw question text (e.g. "What does docker ps do?").
        tokenizer: a HuggingFace tokenizer loaded with
                   AutoTokenizer.from_pretrained(model_name).
                   Each tokenizer ships with its model's correct chat
                   template baked in.

    Returns:
        A string ready to be tokenized and fed to model.generate().

    Example output for Mistral-7B (illustrative):
        "<s>[INST] What does docker ps do? [/INST]"
    """
    messages = [{"role": "user", "content": question}]

    # add_generation_prompt=True appends whatever tokens the model expects
    # to see right before it starts generating. For Mistral that's nothing
    # extra; for Llama-3 it's the assistant header. This makes the helper
    # work correctly across model families.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_chat_pair(question: str, answer: str, tokenizer: Any) -> str:
    """Turn a (question, answer) pair into a full chat string for TRAINING.

    Used in Task #4-5 (training data formatting). The model learns to
    produce the `answer` part when given the formatted `question`.

    Example output for Mistral-7B:
        "<s>[INST] What does docker ps do? [/INST] docker ps lists..."
    """
    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
