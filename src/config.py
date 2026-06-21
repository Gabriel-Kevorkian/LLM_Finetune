"""
Central configuration for the entire project.

WHY a single config file:
    Every script (eval, train, ablation, results compilation) reads its settings
    from here. If we want to switch base model, change the eval set path, or
    tweak a default hyperparameter, we change it ONCE in this file and every
    script picks up the change. This is the opposite of "magic numbers" sprinkled
    across the codebase — those are impossible to keep in sync.

WHAT this file does NOT contain:
    Per-run hyperparameters (LoRA rank, learning rate, etc.) for ablation studies.
    Those live in `configs/*.yaml` so each ablation run has its own frozen,
    versioned config. This file holds DEFAULTS only.
"""

from pathlib import Path


# =============================================================================
# 1. PATHS
# =============================================================================
# Using pathlib.Path (not string concatenation) so paths work on Windows AND
# Linux (Colab) without needing to swap "\" for "/".

# PROJECT_ROOT is computed relative to THIS file's location, so it works no
# matter where you run a script from (project root, scripts/ folder, Colab, etc.)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR        = PROJECT_ROOT / "data"
DATA_RAW_DIR    = DATA_DIR / "raw"          # source documents (read-only)
DATA_EVAL_DIR   = DATA_DIR / "eval"         # locked evaluation set lives here
DATA_TRAIN_DIR  = DATA_DIR / "train"        # training datasets

EVAL_FILE       = DATA_EVAL_DIR / "eval.jsonl"
TRAIN_FILE      = DATA_TRAIN_DIR / "train_1k.jsonl"

RESULTS_DIR     = PROJECT_ROOT / "results"
BASELINE_DIR    = RESULTS_DIR / "baseline"
RUNS_DIR        = RESULTS_DIR / "runs"      # one subfolder per training run
CHARTS_DIR      = RESULTS_DIR / "charts"

MODELS_DIR      = PROJECT_ROOT / "models"
ADAPTERS_DIR    = MODELS_DIR / "adapters"   # LoRA adapter outputs land here

CONFIGS_DIR     = PROJECT_ROOT / "configs"  # YAML configs for each training run


# =============================================================================
# 2. MODEL CONFIG
# =============================================================================
# We use Mistral-7B-v0.3 as the base model.
#
# WHY Mistral-7B-v0.3:
#   - 7 billion parameters: small enough to fine-tune on a free Colab T4 GPU
#     (15GB VRAM) using 4-bit quantization.
#   - Strong general English capabilities (open-source benchmark winner at
#     release time).
#   - Apache 2.0 license — we can publish derivatives freely.
#   - Mistral has good Docker knowledge but not deep — perfect "gap to close"
#     for our fine-tune to show measurable improvement.
#
# Unsloth's prepackaged version (`unsloth/mistral-7b-v0.3-bnb-4bit`) is the
# SAME weights but already quantized and patched for ~2x faster training.
# We'll use it inside the Colab training notebook.
BASE_MODEL_NAME       = "mistralai/Mistral-7B-v0.3"
UNSLOTH_MODEL_NAME    = "unsloth/mistral-7b-v0.3-bnb-4bit"

# Maximum tokens per training example. Most Docker Q&A fits in <512 tokens,
# but we keep 2048 to allow longer answers (multi-step troubleshooting).
# WHY this matters: longer = more VRAM per example. 2048 is the sweet spot
# for a 7B model on a T4.
MAX_SEQ_LENGTH        = 2048

# Use 4-bit quantization during training. WHY:
#   - Reduces VRAM from ~28GB (bfloat16) to ~5GB (4-bit) for a 7B model.
#   - Makes training feasible on a free Colab T4 (15GB).
#   - The PDF mentions <2% quality loss — an excellent tradeoff.
#   - This is what makes "QLoRA" (Quantized LoRA) different from plain LoRA.
LOAD_IN_4BIT          = True


# =============================================================================
# 3. LoRA HYPERPARAMETERS (DEFAULTS)
# =============================================================================
# LoRA = Low-Rank Adaptation.
#
# Quick recap:
#   Fine-tuning a 7B model normally means updating all 7 billion weights.
#   That requires huge VRAM (>80GB) and tons of compute. LoRA instead FREEZES
#   the original weights and INJECTS small trainable matrices alongside specific
#   layers. Only those small matrices get updated during training. Result:
#   we train ~0.1-1% as many parameters with near-identical final quality.
#
# The key knob is the RANK (r):
#   - r controls the size of those injected matrices.
#   - Higher r = more trainable params = more capacity to learn = slower training
#     and higher risk of overfitting on a small dataset.
#   - Lower r = fewer params = faster but maybe under-capacity.
#   - Common values: 4, 8, 16, 32, 64.
#   - Our default is 16 — a balanced starting point. Our ablation will compare
#     r=8, r=16, r=32 to see what's actually best for our 1K-example dataset.
LORA_RANK             = 16

# LORA_ALPHA is a scaling factor for LoRA updates. Conventional wisdom: set it
# equal to LORA_RANK. Don't overthink this on a first project.
LORA_ALPHA            = 16

# Which layers of the model get LoRA adapters attached. For Mistral, these are
# the attention layer projection matrices (q, k, v, o) plus the MLP layers
# (gate, up, down). Targeting all of them tends to give the best results;
# targeting only attention is faster but typically a bit worse.
LORA_TARGET_MODULES   = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",         # MLP / feed-forward
]

# Dropout regularizes training. 0 is fine for our small dataset because we're
# not training long enough to overfit hard. Unsloth performs best with 0.
LORA_DROPOUT          = 0.0


# =============================================================================
# 4. TRAINING HYPERPARAMETERS (DEFAULTS)
# =============================================================================
# These are overridable per-run via configs/*.yaml.

# How aggressively weights update each step. Too high → divergence (loss spikes
# to NaN). Too low → barely learns. 2e-4 is the standard LoRA starting point.
LEARNING_RATE         = 2e-4

# How many full passes through the training data. 3 is a safe LoRA default:
#   - 1 epoch: usually under-trained
#   - 3 epochs: typically the sweet spot
#   - 5+ epochs: often starts overfitting (memorizing training examples
#                instead of generalizing)
NUM_EPOCHS            = 3

# Samples per gradient step. Tiny on Colab T4 because each example may be 2K
# tokens × 7B params worth of activations to hold in VRAM.
PER_DEVICE_BATCH_SIZE = 2

# We accumulate gradients across 4 mini-batches before applying an update.
# Effective batch size = PER_DEVICE_BATCH_SIZE × GRAD_ACCUM_STEPS = 8.
# This gives us the stability of a batch-of-8 with the VRAM cost of batch-of-2.
GRAD_ACCUM_STEPS      = 4

# Learning rate warms up linearly from 0 over this many steps. Stabilizes
# the first few updates which would otherwise be very noisy.
WARMUP_STEPS          = 10

# Reproducibility. Same seed → same random init, same data shuffling, same
# result. CRUCIAL for ablation studies: if r=8 vs r=16 used different seeds,
# we couldn't tell if the difference was from rank or from luck.
SEED                  = 42


# =============================================================================
# 5. EVALUATION CONFIG
# =============================================================================
# Generation settings for running the model over eval questions.
#
# Temperature controls randomness. 0 = deterministic (always picks highest-
# probability next token). >0 = sampled, more varied. For evaluation we want
# 0 so re-running the eval gives the same numbers.
EVAL_TEMPERATURE      = 0.0
EVAL_MAX_NEW_TOKENS   = 512    # cap answer length so eval doesn't hang

# Model used as the LLM-judge. Gemini 2.5 Flash is on Google's free tier
# (15 RPM, 1500 requests/day) — plenty for our 50-200 judge calls per
# ablation run, and zero cost. Fast enough to grade a short answer in <1s.
LLM_JUDGE_MODEL       = "gemini-2.5-flash"


# =============================================================================
# 6. WEIGHTS & BIASES CONFIG
# =============================================================================
# Logs go to https://wandb.ai/<your-username>/<WANDB_PROJECT>.
# We override these via env vars when needed (see .env.example).
WANDB_PROJECT_DEFAULT = "llm-finetune-docker-qa"


# =============================================================================
# 7. CONVENIENCE — make sure output folders exist when imported
# =============================================================================
# Quietly create the folders we'll need to write into. Avoids
# FileNotFoundError on first run.
for _dir in (DATA_RAW_DIR, DATA_EVAL_DIR, DATA_TRAIN_DIR,
             BASELINE_DIR, RUNS_DIR, CHARTS_DIR, ADAPTERS_DIR, CONFIGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
