# Domain-Specific LLM Fine-Tuning with Real Evaluations

A **weekend-scoped** portfolio project that fine-tunes an open-source LLM on a narrow
domain and **proves** the adaptation worked through a rigorous, eval-first methodology.

> **Core principle (unchanged from the 4-week spec):** Build the evaluation suite
> **before** writing any training code. Without a locked benchmark you cannot know
> whether training actually helped.

This is a compressed version of the 3–4 week brief in `LLM_Finetune_Project_Brief.pdf`.
Scope cuts are listed in section [12. What was cut and why](#12-what-was-cut-and-why).

---

## 1. Project Status

| Phase | Status |
|-------|--------|
| Timeline | **One weekend (~16–20 focused hours)** |
| Domain | Tentative: **Technical Support QA** (subject to change before Saturday starts) |
| Hardware | Local: 6 GB VRAM (dev only). Training: Google Colab free T4 (~15 GB). |
| Base model | Mistral-7B-v0.3 with 4-bit QLoRA |
| Evaluation suite | Not built yet |
| Baseline run | Not run yet |
| Fine-tune runs | Not run yet |

---

## 2. Research Question

> Can fine-tuning Mistral-7B on a curated domain-specific dataset measurably improve
> performance on domain tasks compared to the base model — and does LoRA rank matter?

---

## 3. Weekend Plan (Hour-by-Hour)

The pipeline is strictly linear. **Do not start training before `eval.jsonl` is locked.**
Time estimates assume one person, Colab T4 free tier, and zero idle time.

### Saturday — Eval + Baseline + First Fine-Tune

| Block | Hours | Task |
|-------|-------|------|
| Sat AM-1 | ~1h | Lock domain. Set up Python env. Install `unsloth`, `trl`, `peft`, `transformers`, `evaluate`, `wandb`, `google-genai`. Configure HF + Gemini + W&B tokens. |
| Sat AM-2 | ~3h | Hand-build **`data/eval/eval.jsonl`** — **50 verified Q&A pairs** from authoritative source (e.g. official product docs). Lock the file. |
| Sat PM-1 | ~1h | On Colab: run baseline eval (Mistral-7B zero-shot) over the 50 eval examples. Save `results/baseline/`. |
| Sat PM-2 | ~2h | Build **`data/train/train_1k.jsonl`** — 1,000 examples from **different** sources (HF dataset like `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, or GPT-4-generated from doc chunks not used in eval). Deduplicate. Spot-check 30 samples. |
| Sat PM-3 | ~2h | First fine-tune on Colab: LoRA r=16, α=16, lr=2e-4, 3 epochs. Save adapter. |

**End of Saturday goal:** baseline scored, first fine-tuned adapter saved.

### Sunday — Evaluate, One Ablation Axis, Polish

| Block | Hours | Task |
|-------|-------|------|
| Sun AM-1 | ~1h | Eval the Sat fine-tuned adapter (r=16) on the locked eval set. |
| Sun AM-2 | ~3h | **Single ablation axis: LoRA rank.** Train r=8 and r=32 (keeping data, lr, epochs constant). Eval both. |
| Sun PM-1 | ~2h | Compile `results/ablation_table.csv` and a single matplotlib chart (rank vs ROUGE-L). |
| Sun PM-2 | ~2h | **Failure analysis:** pick 5 fine-tuned-model failures, categorize (hallucination / refusal / formatting / OOD), write `results/failure_analysis.md`. |
| Sun PM-3 | ~1h | Update the results table in this README. Make the W&B dashboard public. Push to GitHub. |

**End of Sunday goal:** results table filled, ablation chart in repo, README finalized.

### What's NOT happening this weekend

- Training set size ablation (500/1K/2K) — skipped
- Learning rate ablation — skipped
- Second base model (Qwen) — skipped
- Robustness eval (rephrased questions) — skipped
- Publishing eval dataset / adapter to HuggingFace Hub — optional, push next week
- 3-minute demo video — optional

---

## 4. Folder Architecture

```
LLM_Finetune/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template (HF_TOKEN, GEMINI_API_KEY, WANDB_API_KEY)
├── .gitignore                      # Ignore models/, .env, __pycache__, wandb/
├── LLM_Finetune_Project_Brief.pdf  # Original 4-week project spec
│
├── data/
│   ├── raw/                        # Source documents (read-only)
│   ├── eval/
│   │   └── eval.jsonl              # LOCKED — 50 Q&A pairs
│   └── train/
│       └── train_1k.jsonl          # 1,000 (question, answer) examples
│
├── src/                            # Reusable library code
│   ├── __init__.py
│   ├── config.py                   # Model IDs, hyperparameter defaults
│   ├── data/
│   │   ├── build_eval.py           # Construct eval.jsonl from raw sources
│   │   ├── build_train.py          # Construct train.jsonl from separate sources
│   │   └── format_prompts.py       # tokenizer.apply_chat_template wrappers
│   ├── eval/
│   │   ├── runner.py               # Run a model over the eval set
│   │   ├── metrics.py              # ROUGE-1, ROUGE-L, EM
│   │   └── llm_judge.py            # GPT-4o-mini judge (1–5)
│   ├── training/
│   │   ├── train_lora.py           # Unsloth + TRL SFTTrainer loop
│   │   └── load_model.py           # 4-bit quantization + LoRA wrapping
│   └── inference/
│       └── generate.py             # Load model + adapter, batch generate
│
├── scripts/                        # CLI entry points (numbered = run order)
│   ├── 01_build_eval.py
│   ├── 02_baseline_eval.py
│   ├── 03_build_train.py
│   ├── 04_train.py                 # Takes --config configs/<name>.yaml
│   ├── 05_eval_finetuned.py
│   └── 06_compile_results.py
│
├── notebooks/
│   └── colab_train.ipynb           # Runs on Colab T4 free tier
│
├── configs/                        # One YAML per training run
│   ├── base.yaml                   # Default (r=16, lr=2e-4, 3 epochs, 1K data)
│   ├── ablation_rank_8.yaml
│   └── ablation_rank_32.yaml
│
├── results/
│   ├── baseline/
│   │   ├── results.json
│   │   └── per_example.csv
│   ├── runs/
│   │   ├── r8/
│   │   │   ├── eval_results.json
│   │   │   ├── per_example.csv
│   │   │   └── config.yaml
│   │   ├── r16/
│   │   └── r32/
│   ├── ablation_table.csv
│   ├── failure_analysis.md
│   └── charts/
│       └── rank_vs_rouge.png
│
└── models/                         # gitignored — local adapter cache
    └── adapters/
        ├── r8/
        ├── r16/
        └── r32/
```

### Why this structure (weekend-pragmatic)

- **`src/` vs `scripts/`:** `src/` is reusable; `scripts/` is the runnable pipeline.
  Keeps logic out of CLIs so the Colab notebook can `import src.training` cleanly.
- **Numbered scripts (`01_…` → `06_…`)** make the linear order obvious — important
  when you may revisit at midnight and need to remember what runs next.
- **YAML configs frozen into `results/runs/<name>/`** so every result is reproducible.
- **`models/` gitignored** because adapters (~50–200 MB each) bloat git history.

---

## 5. Evaluation Framework (Built First — Saturday Morning)

Two evaluation dimensions for the weekend version, both in `src/eval/runner.py`:

| Dimension        | What it measures                          | Implementation       |
|------------------|-------------------------------------------|----------------------|
| Factual accuracy | Agreement with reference answer           | ROUGE-1, ROUGE-L, EM |
| LLM-as-Judge     | Nuanced quality vs reference              | GPT-4o-mini, 1–5     |

(Consistency and robustness from the 4-week spec are deferred — see section 12.)

---

## 6. Results Table

Filled in Sunday afternoon. Empty until then.

| Model                       | ROUGE-1 | ROUGE-L | LLM-Judge (1–5) | Notes                |
|-----------------------------|---------|---------|------------------|----------------------|
| Mistral-7B base             | TBD     | TBD     | TBD              | Zero-shot baseline   |
| + LoRA r=8, 1K data         | TBD     | TBD     | TBD              | Lower rank           |
| + LoRA r=16, 1K data        | TBD     | TBD     | TBD              | Default              |
| + LoRA r=32, 1K data        | TBD     | TBD     | TBD              | Higher rank          |

---

## 7. Tooling

| Tool             | Purpose                                  |
|------------------|------------------------------------------|
| Unsloth          | Fast LoRA fine-tuning (2–3× speedup)     |
| HuggingFace PEFT | LoRA implementation                      |
| TRL `SFTTrainer` | Supervised fine-tuning loop              |
| Weights & Biases | Experiment tracking + public dashboard   |
| `evaluate`       | ROUGE, EM                                |
| Google Gemini API| Gemini 2.5 Flash-Lite LLM-as-judge (free tier, 15 RPM) |
| Pandas + Matplotlib | Ablation table + chart                |

---

## 8. Weekend Deliverables

- [ ] GitHub repo with clean code + this README filled in
- [ ] `data/eval/eval.jsonl` (50 examples, locked)
- [ ] Baseline results JSON + CSV
- [ ] Three fine-tuned LoRA adapters (r=8, r=16, r=32)
- [ ] `results/ablation_table.csv` + `rank_vs_rouge.png`
- [ ] `results/failure_analysis.md` (5 categorized failures)
- [ ] Public W&B dashboard
- [ ] (Stretch) Eval dataset + best adapter pushed to HuggingFace Hub

---

## 9. Hardware Plan (6 GB Local + Colab Free)

- **Local (6 GB VRAM):** dataset construction, code editing, eval analysis.
  Cannot train any 7B model, even at 4-bit.
- **Colab free T4 (~15 GB):** every training run. Mount the repo via `git clone`,
  run `notebooks/colab_train.ipynb`, push the adapter back.
- **Time budget per run:** ~1–2 hours for 1K examples × 3 epochs on T4.
  Three ablation runs ≈ 3–6 GPU hours — fits in Colab free tier limits.

---

## 10. Risk Log (Weekend-Specific)

| Risk                                        | Mitigation                                                     |
|---------------------------------------------|----------------------------------------------------------------|
| Colab session times out mid-train           | Save checkpoint every epoch; resume from `models/adapters/`    |
| Eval set takes longer than 3h to build      | Hard-cap at 50; quality > quantity. Stop and move on at 3h     |
| First fine-tune crashes (OOM, dtype, etc.)  | Use Unsloth — it ships sane Colab defaults                     |
| Ablation runs blow through GPU budget       | Cut to r=8 vs r=16 only; drop r=32                             |
| Results show fine-tuning made it WORSE      | This is a legitimate finding — document it in failure analysis |

---

## 11. Anti-Patterns to Avoid

- Starting training before `eval.jsonl` is locked.
- Sourcing training data from the same docs as eval data.
- Using LLM-generated answers as eval ground truth.
- Changing more than one hyperparameter per ablation run.
- Reporting only successes — the failure analysis carries half the project's signal.
- Hardcoding `[INST]` tokens instead of using `tokenizer.apply_chat_template()`.

---

## 12. What Was Cut and Why

Compared to the 4-week brief in the PDF, the following were dropped to fit a weekend.
List them in the README so a reviewer sees the tradeoffs explicitly — that itself
demonstrates engineering judgment.

| Cut                                          | Why it's safe to cut for v1                                 |
|----------------------------------------------|-------------------------------------------------------------|
| Eval set: 100–300 → **50**                   | Smaller N is still credible for a portfolio project; can grow later |
| Training set: 2K–5K → **1K**                 | Ablation question is still answerable with 1K               |
| Ablation axes: 3 → **1 (LoRA rank only)**    | One controlled axis still produces an interview talking point |
| Second base model (Qwen)                     | Not required to answer the research question                |
| Consistency + Robustness eval dimensions     | Nice-to-have; ROUGE + LLM-judge is the strong core          |
| HuggingFace Hub publishing                   | Can do in a follow-up; doesn't gate the engineering result  |
| 3-minute demo video                          | Optional polish                                             |

If extra time appears: re-add (in order) — consistency eval, training-size ablation,
demo video, HF Hub publishing.

---

## 13. References

- Project brief: `LLM_Finetune_Project_Brief.pdf`
- Unsloth: https://github.com/unslothai/unsloth
- HuggingFace PEFT: https://huggingface.co/docs/peft
- TRL: https://huggingface.co/docs/trl

---

*Next action: confirm the domain choice, then begin Saturday AM-1 — set up the environment and start building `data/eval/eval.jsonl`.*
