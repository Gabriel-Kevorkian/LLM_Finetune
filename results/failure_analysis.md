# Failure Analysis — Fine-Tuned Mistral-7B (Docker Q&A)

Qualitative analysis of where the best fine-tuned adapter (**r=32**, judge 3.24/5)
still fails, on the locked 50-question eval set (`data/eval/eval.jsonl`).
Goal: turn the 17 judge≤2 examples into a small, actionable taxonomy that points
at *root causes* and *concrete fixes* — not just a list of wrong answers.

**Method:** I read every `judge_score == 2` row in `results/runs/r32/per_example.csv`
(17 of 50), grouped them by failure *mode*, and cross-checked each example against
the r=8 and r=16 runs to separate **decoding/generation problems** (which vary run
to run) from **genuine knowledge gaps** (which are wrong at every rank).

## Headline numbers

| Signal | Value |
|--------|-------|
| Judge score distribution (r=32) | 2→17, 3→12, 4→13, 5→8 |
| Predictions with runaway repetition (≥3× repeated sentence or >1400 chars) | **14 / 50** |
| Avg reference length vs avg prediction length | 373 chars vs **932 chars** (~2.5×) |
| Runaway-repetition share of the 17 judge=2 failures | 7 / 17 |
| Worst category (judge=2 count) | `dockerfile` — 8 of 17 |
| Exact-match | 0 / 50 (expected: free-form answers, EM is a weak signal here) |

The single biggest problem is **generation dynamics, not knowledge**: the model
frequently *knows the correct first sentence* and then cannot stop.

---

## Failure taxonomy (5 modes)

### 1. Degenerate repetition loop — *most common (≈14/50)*
**Example `docker-003` — "What does the FROM instruction do?"** (easy)
The first sentence is correct (`FROM ... sets the base image`), then the model
loops a template and starts **inventing flags that don't exist**:

> "...the image used to create the container when `docker build` is run with the
> `--no-target` option ... the `--no-network` option ... the `--build-arg` option..."

`--no-target` / `--no-network` are not real `docker build` flags. The loop both
wastes the whole 512-token budget and manufactures false facts. Severity: **high**
(an easy question scored 2 purely because of decoding).

### 2. Semantic drift inside the loop
**Example `docker-017` — "What does EXPOSE do, and does it publish a port?"** (easy)
Starts perfectly (`EXPOSE does not publish any ports... documents what ports the
container listens on`), but the repetition loop *drifts off-topic*:

> "It is for the reverse engineer to document... It is for the hacker to try to
> break into the application by guessing the port numbers." (repeated ~20×)

The correct answer was within reach — it even nails the `-P` vs `-p` distinction
in the reference's territory — but degenerates into nonsense. Severity: **high**.

### 3. Confidently wrong on a security question — *knowledge gap, wrong at every rank*
**Example `docker-047` — "How do you pass a secret to a build step without it
landing in image layers?"** (security, hard) — **judge 2 at r=8, r=16, AND r=32.**
The model recommends the **exact anti-pattern** the question warns against:

> "You can use `--build-arg` ... `RUN echo $API_KEY` ... The argument is not
> stored in the built image."

This is false and dangerous: `--build-arg` values **are** recoverable from
`docker history` and layer metadata. The correct answer (BuildKit
`--secret` + `RUN --mount=type=secret`) never appears. Because it fails at all
three ranks, more LoRA capacity won't fix it — it's a **training-data gap**.
Severity: **critical** (confidently wrong security advice).

### 4. Fabricated mechanism — *knowledge gap, wrong at every rank*
**Example `docker-029` — "What does depends_on do, and what does it NOT
guarantee?"** (compose, medium) — judge 2 at r=16 and r=32.
The model invents Compose behavior that does not exist:

> "...it is not a strong constraint, and the engine may reorder things if it
> thinks it is better (for example ... to minimize downtime)."

Compose does **not** reorder `depends_on`. Worse, it misses the actual answer the
question is fishing for: `depends_on` waits for *start*, not *readiness*; you need
a `healthcheck` + `condition: service_healthy`. Right instinct ("doesn't guarantee
much"), fabricated justification, missing real fix. Severity: **high**.

### 5. Muddled / self-contradictory reasoning — *knowledge gap, wrong at every rank*
**Example `docker-045` — "Difference between CMD and ENTRYPOINT, and how do they
interact?"** (dockerfile, medium) — judge 2 at all three ranks.
The general gist is present (CMD = default args to ENTRYPOINT) but buried in
contradictions and a garbled example:

> "The `ENTRYPOINT` command can be overridden... but the `CMD` command cannot be
> overridden." (false — CMD is the *easily* overridable part)
> "...if the `ENTRYPOINT` is `sh -c` and the `CMD` is `ls -lah /`..."

It never states the key rule (in exec form, CMD is *appended* to ENTRYPOINT as
arguments). Partial knowledge, no clean mental model. Severity: **medium**.

---

## Root-cause hypotheses

1. **Inference decoding has no repetition control (drives modes 1–2).**
   Eval uses greedy decoding (`do_sample=False`, `temperature=0`,
   `max_new_tokens=512`) with **no `repetition_penalty` and no
   `no_repeat_ngram_size`**. Greedy + a weak EOS signal is the classic recipe for
   loops. The length gap (932 vs 373 chars) confirms the model rarely emits a
   timely stop token.

2. **Training answers are long and weakly terminated (amplifies #1).**
   Training data is 1K Stack Overflow `[docker]` answers — verbose, multi-paragraph,
   and not curated to end crisply. Three epochs over that teaches "keep
   explaining," not "answer tightly and stop." The model learned Docker *content*
   but not Docker-answer *shape*.

3. **Genuine subtopic gaps (modes 3–5) are rank-independent.**
   `docker-047`, `docker-029`, `docker-045` score 2 at **every** rank. These are
   advanced, lower-frequency subtopics (build secrets / BuildKit, Compose readiness
   vs start order, ENTRYPOINT⊕CMD exec-form semantics) that are simply
   under-represented in 1K general SO answers. Adding LoRA capacity (r=8→32) does
   not help — only better/targeted **data** will.

4. **Single-run judge noise on a 50-item set (a caveat, not a failure).**
   `docker-003`/`docker-017` scored 4/3 at lower ranks but 2 at r=32 — the
   repetition trigger is partly stochastic and the judge penalizes it
   inconsistently. On 50 questions, ±1 judge point per item is real variance;
   treat individual scores as directional, not exact.

## Recommended remediations (prioritized)

| # | Fix | Cost | Expected impact |
|---|-----|------|-----------------|
| 1 | **Add `repetition_penalty≈1.2` and `no_repeat_ngram_size=3` at inference** and re-run eval | ~free (no retrain, 1 eval pass) | Directly targets modes 1–2 (the 7 runaway judge=2 cases + likely lifts several 3s) |
| 2 | **Curate training answers to end crisply** (trim to the answer, ensure each ends with `eos_token`) | medium (data prep + 1 retrain) | Strengthens EOS, reduces loops at the source |
| 3 | **Targeted data augmentation** for the rank-independent gaps: BuildKit secrets, Compose `healthcheck`/`service_healthy`, ENTRYPOINT/CMD exec-form | medium | Only thing that moves modes 3–5 |
| 4 | Keep **r=32** as the shipped adapter | done | Best on every metric; rank already maxed for this data |

**Status of fix #1:** the repetition controls are now wired into the eval code —
`config.EVAL_REPETITION_PENALTY = 1.2` and `EVAL_NO_REPEAT_NGRAM_SIZE = 3`, read by
*both* `scripts/02_baseline_eval.py` and `scripts/05_eval_finetuned.py`. They are
**not yet reflected in the committed numbers**: the results under `results/` were
all produced with pure greedy decoding (pre-fix) and remain mutually comparable.
Validating the fix would require one fresh Colab pass over *both* the baseline and
the r=32 adapter (to keep the comparison fair) — deliberately deferred here, so the
improvement is a documented, ready-to-run hypothesis rather than a measured result.

**Bottom line:** the fine-tune clearly worked (judge 2.28 → 3.24, ROUGE-1 +131%),
but the remaining failures split cleanly into a cheap decoding fix (repetition,
~½ of the worst cases) and a data-quality investment (a handful of advanced
subtopics that no amount of LoRA rank will fix). The single highest-value next
step is #1 — a one-line generation-config change and a re-eval, no retraining.

*Source data: `results/runs/{r8,r16,r32}/per_example.csv`; aggregate metrics in
`results/ablation_table.csv`; trend chart in `results/charts/rank_vs_rouge.png`.*
