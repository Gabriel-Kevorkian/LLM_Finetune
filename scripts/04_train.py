"""
CLI: fine-tune Mistral-7B with LoRA on data/train/train_1k.jsonl.

USAGE (on Colab T4, NOT on the local 6 GB box):
    python scripts/04_train.py --config configs/base.yaml
    python scripts/04_train.py --config configs/base.yaml --wandb

WHAT IT DOES:
    1. Parses --config <path>.yaml into a TrainConfig dataclass.
    2. (Optional) toggles W&B logging when --wandb is passed.
    3. Calls src.training.train_lora.run_training to do the actual work.
    4. Prints the final loss + adapter location.

LOCAL-FAIL NOTE:
    Running this on the user's Windows box will crash on `import unsloth`
    (no CUDA + no bnb). That's expected — see notebooks/colab_train.ipynb.
"""

import argparse
import sys
from pathlib import Path

# Make `src.*` importable when running this script from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 stdout so progress bars / ≥ don't crash on Colab logs piped
# through Jupyter (cp1252 default on Windows; Jupyter is fine but be defensive).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config as project_config
from src.training.train_lora import load_train_config, run_training


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a configs/*.yaml training config (e.g. configs/base.yaml)",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (overrides wandb_enabled in YAML).",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default=project_config.UNSLOTH_MODEL_NAME,
        help="HF model ID for the base model. Defaults to the Unsloth pre-"
             "quantized Mistral-7B variant (fastest download).",
    )
    args = p.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found at {args.config}")
        return 1

    train_cfg = load_train_config(args.config)

    # CLI flag overrides YAML — handy in the Colab notebook where you may
    # toggle wandb without editing the YAML file.
    if args.wandb:
        train_cfg.wandb_enabled = True

    print("=" * 70)
    print(f"Fine-tune run: {train_cfg.run_name}")
    print("=" * 70)
    print(f"  base model:  {args.base_model}")
    print(f"  train file:  {project_config.TRAIN_FILE}")
    print(f"  rank/alpha:  {train_cfg.lora_rank}/{train_cfg.lora_alpha}")
    print(f"  lr:          {train_cfg.learning_rate}")
    print(f"  epochs:      {train_cfg.num_epochs}")
    print(f"  batch:       {train_cfg.per_device_batch_size} "
          f"× {train_cfg.grad_accum_steps} grad-accum "
          f"= {train_cfg.per_device_batch_size * train_cfg.grad_accum_steps} effective")
    print(f"  seq_len:     {train_cfg.max_seq_length}")
    print(f"  wandb:       {train_cfg.wandb_enabled}")
    print(f"  seed:        {train_cfg.seed}")
    print()

    summary = run_training(
        config        = train_cfg,
        train_file    = project_config.TRAIN_FILE,
        base_model_id = args.base_model,
        adapters_dir  = project_config.ADAPTERS_DIR,
        results_dir   = project_config.RUNS_DIR,
        yaml_path     = args.config,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  final loss:    {summary['final_loss']:.4f}")
    print(f"  steps:         {summary['global_step']}")
    print(f"  adapter:       {summary['adapter_path']}")
    print(f"  run dir:       {project_config.RUNS_DIR / train_cfg.run_name}")
    print()
    print("NEXT STEP: run scripts/05_eval_finetuned.py to score this adapter "
          "against data/eval/eval.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
