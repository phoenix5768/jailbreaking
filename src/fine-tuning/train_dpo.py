#!/usr/bin/env python3
"""
DPO-LoRA training on preference data with epoch-level checkpoints and metric export.

Expected input JSONL format:
  {"prompt": "...", "chosen": "...", "rejected": "..."}

Default behavior:
  - trains for 5 epochs
  - saves one checkpoint at the end of each epoch
  - exports trainer logs as JSON/CSV
  - creates loss and DPO metric plots when matplotlib/pandas are available
  - saves the final adapter under <output_dir>/final_adapter

Optional:
  --init_adapter /path/to/sft_adapter
      Continue training from an existing PEFT adapter, e.g. SFT -> DPO.
      If omitted, a fresh DPO LoRA adapter is trained from the base model.

  --eval_file /path/to/valid.jsonl
      Evaluate once per epoch on a separate DPO validation file.

  --eval_ratio 0.1
      If --eval_file is not provided, split this fraction from --train_file for eval.
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and safety-aligned assistant. Refuse unsafe requests "
    "briefly and offer safe alternatives. For benign requests, be helpful."
)

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
            rows.append(obj)
    return rows


def validate_rows(rows: List[Dict], dataset_name: str = "dataset") -> None:
    required = {"prompt", "chosen", "rejected"}
    if not rows:
        raise ValueError(f"{dataset_name} is empty.")

    for i, row in enumerate(rows):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"{dataset_name} row {i} is missing required keys: {sorted(missing)}")

        for key in required:
            if not isinstance(row[key], str):
                raise ValueError(f"{dataset_name} row {i} key '{key}' must be a string.")
            if not row[key].strip():
                raise ValueError(f"{dataset_name} row {i} key '{key}' is empty.")

        if row["chosen"].strip() == row["rejected"].strip():
            raise ValueError(f"{dataset_name} row {i} has identical chosen and rejected responses.")


def split_train_eval_rows(
    rows: List[Dict],
    eval_ratio: float,
    shuffle: bool,
    seed: int,
) -> Tuple[List[Dict], Optional[List[Dict]]]:
    if eval_ratio <= 0:
        return rows, None
    if eval_ratio >= 1:
        raise ValueError("--eval_ratio must be lower than 1.0")
    if len(rows) < 2:
        raise ValueError("Need at least 2 rows to create a train/eval split.")

    rows_for_split = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows_for_split)

    eval_size = max(1, round(len(rows_for_split) * eval_ratio))
    eval_rows = rows_for_split[:eval_size]
    train_rows = rows_for_split[eval_size:]

    if not train_rows:
        raise ValueError("Train split became empty. Use a smaller --eval_ratio.")

    return train_rows, eval_rows


def to_conversational_dpo(rows: List[Dict], system_prompt: str, shuffle: bool, seed: int) -> Dataset:
    examples = []
    for row in rows:
        examples.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["prompt"].strip()},
                ],
                "chosen": [{"role": "assistant", "content": row["chosen"].strip()}],
                "rejected": [{"role": "assistant", "content": row["rejected"].strip()}],
            }
        )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)

    return Dataset.from_list(examples)


def count_trainable_params(model) -> Dict[str, float]:
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    pct = 100 * trainable / total if total else 0
    return {"trainable": trainable, "total": total, "percent": pct}


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def compute_schedule_info(dataset_size: int, batch_size: int, grad_accum: int, epochs: float) -> Dict[str, float]:
    world_size = get_world_size()
    effective_batch = batch_size * grad_accum * world_size
    steps_per_epoch = math.ceil(dataset_size / effective_batch)
    total_optimizer_steps = math.ceil(steps_per_epoch * epochs)

    return {
        "world_size": world_size,
        "effective_batch_size": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "checkpoint_strategy": "epoch",
        "expected_epoch_checkpoints": math.ceil(epochs),
    }


def load_trainable_peft_model(base_model_id: str, adapter_path: str, local_files_only: bool):
    base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=local_files_only,
        device_map=None,
    )

    base.config.use_cache = False
    base.gradient_checkpointing_enable()

    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=True,
    )

    return model


def make_dpo_config(config_kwargs: Dict):
    """
    Transformers renamed evaluation_strategy to eval_strategy in newer versions.
    This helper keeps the script usable across both variants.
    """
    try:
        return DPOConfig(**config_kwargs)
    except TypeError as e:
        msg = str(e)
        if "eval_strategy" in msg and "eval_strategy" in config_kwargs:
            config_kwargs = dict(config_kwargs)
            config_kwargs["evaluation_strategy"] = config_kwargs.pop("eval_strategy")
            return DPOConfig(**config_kwargs)
        if "evaluation_strategy" in msg and "evaluation_strategy" in config_kwargs:
            config_kwargs = dict(config_kwargs)
            config_kwargs["eval_strategy"] = config_kwargs.pop("evaluation_strategy")
            return DPOConfig(**config_kwargs)
        raise


def export_log_history(trainer, output_dir: Path) -> None:
    log_history = trainer.state.log_history

    json_path = output_dir / "log_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)
    print("Saved log history JSON:", json_path)

    csv_path = output_dir / "log_history.csv"
    all_keys = sorted({key for row in log_history for key in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in log_history:
            writer.writerow(row)
    print("Saved log history CSV:", csv_path)


def save_epoch_summary_and_plots(output_dir: Path) -> None:
    """
    Creates:
      - epoch_metrics_summary.csv
      - loss_curve.png
      - dpo_reward_margin.png, if rewards/margins exists
      - dpo_reward_accuracy.png, if rewards/accuracies exists

    If pandas or matplotlib are missing, training still finishes successfully.
    """
    csv_path = output_dir / "log_history.csv"
    if not csv_path.exists():
        print("No log_history.csv found; skipping plots.")
        return

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Could not import pandas/matplotlib; skipping plots: {e}")
        return

    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns:
        print("Log history has no epoch column; skipping plots.")
        return

    # Epoch-level compact summary.
    max_epoch = int(math.ceil(float(df["epoch"].dropna().max()))) if df["epoch"].notna().any() else 0
    summary_rows = []
    for epoch in range(1, max_epoch + 1):
        lower = epoch - 1
        upper = epoch
        train_rows = df[df.get("loss").notna() & (df["epoch"] > lower) & (df["epoch"] <= upper)] if "loss" in df.columns else pd.DataFrame()
        eval_rows = df[df.get("eval_loss").notna() & (df["epoch"].round(6) <= upper) & (df["epoch"].round(6) > lower)] if "eval_loss" in df.columns else pd.DataFrame()

        row = {"epoch": epoch}
        if not train_rows.empty:
            row["train_loss_mean"] = train_rows["loss"].mean()
            row["train_loss_last"] = train_rows["loss"].iloc[-1]
        if not eval_rows.empty:
            row["eval_loss"] = eval_rows["eval_loss"].iloc[-1]

        # Common DPO metrics from TRL. Keep whichever exist in your version.
        for metric in [
            "rewards/chosen",
            "rewards/rejected",
            "rewards/margins",
            "rewards/accuracies",
            "eval_rewards/chosen",
            "eval_rewards/rejected",
            "eval_rewards/margins",
            "eval_rewards/accuracies",
        ]:
            if metric in df.columns:
                metric_rows = df[df[metric].notna() & (df["epoch"] > lower) & (df["epoch"] <= upper)]
                if not metric_rows.empty:
                    row[metric.replace("/", "_")] = metric_rows[metric].iloc[-1]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "epoch_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("Saved epoch metrics summary:", summary_path)

    # Loss curve.
    plt.figure(figsize=(8, 5))
    plotted = False
    if "loss" in df.columns:
        train_df = df[df["loss"].notna()]
        if not train_df.empty:
            plt.plot(train_df["epoch"], train_df["loss"], marker="o", label="Training loss")
            plotted = True
    if "eval_loss" in df.columns:
        eval_df = df[df["eval_loss"].notna()]
        if not eval_df.empty:
            plt.plot(eval_df["epoch"], eval_df["eval_loss"], marker="o", label="Validation loss")
            plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("DPO-LoRA training and validation loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = output_dir / "loss_curve.png"
        plt.savefig(path, dpi=300)
        print("Saved loss curve:", path)
    plt.close()

    # DPO reward margin curve.
    margin_cols = [col for col in ["rewards/margins", "eval_rewards/margins"] if col in df.columns]
    if margin_cols:
        plt.figure(figsize=(8, 5))
        plotted = False
        for col in margin_cols:
            cur = df[df[col].notna()]
            if not cur.empty:
                label = "Training reward margin" if col == "rewards/margins" else "Validation reward margin"
                plt.plot(cur["epoch"], cur[col], marker="o", label=label)
                plotted = True
        if plotted:
            plt.xlabel("Epoch")
            plt.ylabel("Reward margin")
            plt.title("DPO chosen-vs-rejected reward margin")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            path = output_dir / "dpo_reward_margin.png"
            plt.savefig(path, dpi=300)
            print("Saved reward margin plot:", path)
        plt.close()

    # DPO preference accuracy curve.
    acc_cols = [col for col in ["rewards/accuracies", "eval_rewards/accuracies"] if col in df.columns]
    if acc_cols:
        plt.figure(figsize=(8, 5))
        plotted = False
        for col in acc_cols:
            cur = df[df[col].notna()]
            if not cur.empty:
                label = "Training preference accuracy" if col == "rewards/accuracies" else "Validation preference accuracy"
                values = cur[col]
                # TRL usually logs this as 0..1. Convert to percent for readability.
                if values.max() <= 1.0:
                    values = values * 100
                plt.plot(cur["epoch"], values, marker="o", label=label)
                plotted = True
        if plotted:
            plt.xlabel("Epoch")
            plt.ylabel("Preference accuracy (%)")
            plt.title("DPO preference accuracy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            path = output_dir / "dpo_preference_accuracy.png"
            plt.savefig(path, dpi=300)
            print("Saved preference accuracy plot:", path)
        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", required=True, help="Base model path or HF id.")
    parser.add_argument("--train_file", required=True, help="JSONL with prompt/chosen/rejected.")
    parser.add_argument("--output_dir", required=True, help="Where to save DPO adapter checkpoints.")

    parser.add_argument(
        "--init_adapter",
        default=None,
        help=(
            "Optional existing PEFT adapter to continue from, e.g. SFT adapter. "
            "If omitted, trains a fresh DPO LoRA adapter from the base model."
        ),
    )

    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="sigmoid")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.0)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--local_files_only", action="store_true", default=True)

    args = parser.parse_args()

    if args.eval_ratio < 0 or args.eval_ratio >= 1:
        raise ValueError("--eval_ratio must be in the range [0, 1).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=" * 80)
    print("DPO-LoRA training")
    print("Model:", args.model_id)
    print("Train file:", args.train_file)
    print("Eval file:", args.eval_file)
    print("Output dir:", args.output_dir)
    print("Epochs:", args.epochs)
    print("Checkpoint strategy: save once per epoch")
    print("=" * 80)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Reading DPO dataset...")
    all_rows = read_jsonl(args.train_file)
    validate_rows(all_rows, dataset_name="train_file")

    eval_rows = None
    if args.eval_file:
        eval_rows = read_jsonl(args.eval_file)
        validate_rows(eval_rows, dataset_name="eval_file")
        train_rows = all_rows
    else:
        train_rows, eval_rows = split_train_eval_rows(
            rows=all_rows,
            eval_ratio=args.eval_ratio,
            shuffle=not args.no_shuffle,
            seed=args.seed,
        )

    train_dataset = to_conversational_dpo(
        rows=train_rows,
        system_prompt=args.system_prompt,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    eval_dataset = None
    if eval_rows:
        eval_dataset = to_conversational_dpo(
            rows=eval_rows,
            system_prompt=args.system_prompt,
            shuffle=False,
            seed=args.seed,
        )

    schedule_info = compute_schedule_info(
        dataset_size=len(train_dataset),
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
    )

    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size:", len(eval_dataset) if eval_dataset is not None else 0)
    print("World size:", schedule_info["world_size"])
    print("Effective batch size:", schedule_info["effective_batch_size"])
    print("Steps per epoch:", schedule_info["steps_per_epoch"])
    print("Total optimizer steps:", schedule_info["total_optimizer_steps"])

    lora_config: Optional[LoraConfig] = None
    model = None

    if args.init_adapter:
        print("Continuing from existing adapter:", args.init_adapter)
        model = load_trainable_peft_model(
            base_model_id=args.model_id,
            adapter_path=args.init_adapter,
            local_files_only=args.local_files_only,
        )
    else:
        print("Training a fresh DPO LoRA adapter from the base model.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        )

    has_eval = eval_dataset is not None
    config_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        eval_strategy="epoch" if has_eval else "no",
        do_eval=has_eval,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_length,
        beta=args.beta,
        loss_type=args.loss_type,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "local_files_only": args.local_files_only,
            "device_map": None, 
        } if model is None else None,
    )

    # Remove None values because some TrainingArguments/DPOConfig versions reject them.
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    training_args = make_dpo_config(config_kwargs)

    trainer_kwargs = dict(
        model=args.model_id if model is None else model,
        args=training_args,
        train_dataset=train_dataset,
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if lora_config is not None:
        trainer_kwargs["peft_config"] = lora_config

    try:
        trainer = DPOTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        print("DPOTrainer did not accept processing_class; retrying with tokenizer=...")
        trainer = DPOTrainer(**trainer_kwargs, tokenizer=tokenizer)

    if hasattr(trainer.model, "config"):
        trainer.model.config.use_cache = False

    params = count_trainable_params(trainer.model)
    print(
        f"Trainable parameters: {params['trainable']:,} / {params['total']:,} "
        f"({params['percent']:.4f}%)"
    )

    metadata = {
        "method": "DPO-LoRA",
        "base_model": args.model_id,
        "init_adapter": args.init_adapter,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "eval_ratio": args.eval_ratio,
        "original_train_file_size": len(all_rows),
        "train_dataset_size": len(train_dataset),
        "eval_dataset_size": len(eval_dataset) if eval_dataset is not None else 0,
        "system_prompt": args.system_prompt,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": schedule_info["effective_batch_size"],
        "world_size": schedule_info["world_size"],
        "steps_per_epoch": schedule_info["steps_per_epoch"],
        "total_optimizer_steps": schedule_info["total_optimizer_steps"],
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": LORA_TARGET_MODULES,
        },
        "seed": args.seed,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "trainable_parameters": params,
    }

    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("Run metadata saved to:", metadata_path)

    print("Starting DPO training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    if eval_dataset is not None:
        print("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    final_dir = output_dir / "final_adapter"
    print("Saving final adapter to:", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    export_log_history(trainer, output_dir)
    save_epoch_summary_and_plots(output_dir)

    print("=" * 80)
    print("DPO training completed.")
    print("Intermediate checkpoints saved under:", output_dir)
    print("Final adapter saved to:", final_dir)
    print("Logs and plots saved under:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
