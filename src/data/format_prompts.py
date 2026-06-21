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

THE BASE-MODEL CHAT-TEMPLATE GOTCHA:
    Mistral-7B-v0.3 is the BASE model. It ships WITHOUT a chat template —
    only the Instruct variant has one. Calling apply_chat_template() on a
    bare base tokenizer raises:
        ValueError: Cannot use chat template functions because
        tokenizer.chat_template is not set
    Our fix: ensure_chat_template() installs Mistral's canonical
    [INST]...[/INST] template if the tokenizer doesn't already have one.
    Safe no-op if a template is already set (e.g. if you swap to the
    Instruct variant later, or to Qwen, etc.).
"""

from __future__ import annotations

from typing import Any


# Canonical Mistral chat template (matches mistralai/Mistral-7B-Instruct-v0.3).
# This is the Jinja2 string that the HuggingFace tokenizer executes when
# you call apply_chat_template().
#
# Produces:
#   single user msg with add_generation_prompt=True:
#     "<s> [INST] What does docker ps do? [/INST]"
#   user + assistant pair:
#     "<s> [INST] What does docker ps do? [/INST] docker ps lists ...</s>"
#
# The `{{- bos_token }}` adds Mistral's `<s>` automatically — DO NOT prepend
# it yourself when tokenizing or you'll get a doubled BOS.
MISTRAL_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
        "{%- set system_message = messages[0]['content'] %}"
        "{%- set loop_messages = messages[1:] %}"
    "{%- else %}"
        "{%- set loop_messages = messages %}"
    "{%- endif %}"
    "{{- bos_token }}"
    "{%- for message in loop_messages %}"
        "{%- if message['role'] == 'user' %}"
            "{%- if loop.first and system_message is defined %}"
                "{{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}"
            "{%- else %}"
                "{{- ' [INST] ' + message['content'] + ' [/INST]' }}"
            "{%- endif %}"
        "{%- elif message['role'] == 'assistant' %}"
            "{{- ' ' + message['content'] + eos_token }}"
        "{%- endif %}"
    "{%- endfor %}"
)


def ensure_chat_template(tokenizer: Any) -> None:
    """Install Mistral's chat template on the tokenizer if it has none.

    Idempotent — if the tokenizer already has a chat_template (e.g. you
    loaded Mistral-7B-Instruct-v0.3, or Qwen, or Llama-3), this does
    nothing and we keep the model's native format. The fallback only kicks
    in for bare base-model tokenizers.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE


def format_chat(question: str, tokenizer: Any) -> str:
    """Turn a single user question into the model's chat template.

    Args:
        question:  the raw question text (e.g. "What does docker ps do?").
        tokenizer: a HuggingFace tokenizer loaded with
                   AutoTokenizer.from_pretrained(model_name).

    Returns:
        A string ready to be tokenized and fed to model.generate().

    Example output for Mistral:
        "<s> [INST] What does docker ps do? [/INST]"
    """
    ensure_chat_template(tokenizer)

    messages = [{"role": "user", "content": question}]

    # add_generation_prompt=True appends whatever tokens the model expects
    # to see right before it starts generating. For Mistral that's nothing
    # extra (the prompt ends right after [/INST]); for Llama-3 it would
    # add the assistant header. apply_chat_template handles all of that.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_chat_pair(question: str, answer: str, tokenizer: Any) -> str:
    """Turn a (question, answer) pair into a full chat string for TRAINING.

    Used in Task #4-5 (training data formatting). The model learns to
    produce the `answer` part when given the formatted `question`.

    Example output for Mistral:
        "<s> [INST] What does docker ps do? [/INST] docker ps lists ...</s>"
    """
    ensure_chat_template(tokenizer)

    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
