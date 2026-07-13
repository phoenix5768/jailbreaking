import argparse
import csv
import inspect
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


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
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            if "prompt" not in row:
                raise ValueError(f"Line {line_no} is missing required field 'prompt'.")
            if "answer" not in row:
                raise ValueError(f"Line {line_no} is missing required field 'answer'.")
            if "type" not in row:
                raise ValueError(
                    f"Line {line_no} is missing required field 'type'. "
                    "Expected type='refusal' or type='safe'."
                )

            ex_type = str(row["type"]).lower()
            if ex_type not in {"refusal", "safe"}:
                raise ValueError(
                    f"Line {line_no} has unsupported type={row['type']!r}. "
                    "Expected 'refusal' or 'safe'."
                )

            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError(f"Line {line_no} has an empty or invalid prompt.")
            if not isinstance(row["answer"], str) or not row["answer"].strip():
                raise ValueError(f"Line {line_no} has an empty or invalid answer.")

            rows.append(
                {
                    "type": ex_type,
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                }
            )

    if not rows:
        raise ValueError(f"No examples found in {path}")

    return rows


def count_types(rows: List[Dict]) -> Dict[str, int]:
    counts = {}
    for row in rows:
        t = row.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def split_train_eval(
    rows: List[Dict],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """Stratified split by the 'type' field, preserving refusal/safe balance."""
    if eval_ratio <= 0:
        return rows, None
    if eval_ratio >= 1:
        raise ValueError("--eval_ratio must be between 0 and 1, for example 0.1")

    rng = random.Random(seed)
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(row.get("type", "unknown"), []).append(row)

    train_rows = []
    eval_rows = []

    for _, group in groups.items():
        group = list(group)
        rng.shuffle(group)

        if len(group) == 1:
            # Keep singleton groups in train so that training never loses a type entirely.
            train_rows.extend(group)
            continue

        n_eval = max(1, int(round(len(group) * eval_ratio)))
        n_eval = min(n_eval, len(group) - 1)

        eval_rows.extend(group[:n_eval])
        train_rows.extend(group[n_eval:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    if not eval_rows:
        return train_rows, None

    return train_rows, eval_rows


class MixedSFTDataset(Dataset):
    """
    Expected JSONL format, one object per line:

    Refusal example:
    {"type": "refusal", "prompt": "...", "answer": "safe refusal answer"}

    Safe/helpful example:
    {"type": "safe", "prompt": "...", "answer": "helpful answer"}

    The loss is computed only on the assistant answer. The system/user prompt
    tokens are masked with -100.
    """

    def __init__(
        self,
        rows: List[Dict],
        tokenizer,
        max_length: int = 2048,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        user_prompt = row["prompt"]
        assistant_answer = row["answer"]

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize prompt and answer separately so that prompt truncation does
        # not accidentally remove the whole assistant answer from the loss.
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        answer_ids = self.tokenizer(
            assistant_answer + self.tokenizer.eos_token,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # Keep the answer whenever possible. If the sequence is too long,
        # truncate the prompt from the left and keep its end, which contains
        # the assistant generation marker.
        if len(prompt_ids) + len(answer_ids) > self.max_length:
            max_prompt_len = self.max_length - len(answer_ids)

            if max_prompt_len <= 0:
                # Extremely long answer. Rare, but keep a trainable target.
                prompt_ids = []
                answer_ids = answer_ids[: self.max_length]
            else:
                prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + answer_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = max_len - cur_len

            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def compute_schedule_info(
    num_train_examples: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
) -> Dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = batch_size * grad_accum * world_size
    steps_per_epoch = math.ceil(num_train_examples / effective_batch_size)
    total_optimization_steps = math.ceil(steps_per_epoch * epochs)

    return {
        "world_size": world_size,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_optimization_steps": total_optimization_steps,
    }


def write_run_metadata(
    output_dir: str,
    args,
    num_total_examples: int,
    train_type_counts: Dict[str, int],
    eval_type_counts: Optional[Dict[str, int]],
    schedule_info: Dict,
):
    metadata = {
        "model_id": args.model_id,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "eval_ratio": args.eval_ratio,
        "output_dir": args.output_dir,
        "num_total_examples_loaded": num_total_examples,
        "num_train_examples": sum(train_type_counts.values()),
        "num_eval_examples": sum(eval_type_counts.values()) if eval_type_counts else 0,
        "train_type_counts": train_type_counts,
        "eval_type_counts": eval_type_counts,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": schedule_info["effective_batch_size"],
        "world_size": schedule_info["world_size"],
        "steps_per_epoch": schedule_info["steps_per_epoch"],
        "total_optimization_steps": schedule_info["total_optimization_steps"],
        "save_strategy": "epoch",
        "checkpoint_epochs": [i for i in range(1, int(math.floor(args.epochs)) + 1)],
        "save_total_limit": args.save_total_limit,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": LORA_TARGET_MODULES,
        },
        "system_prompt": args.system_prompt,
        "seed": args.seed,
    }

    path = os.path.join(output_dir, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Run metadata saved to: {path}")


def save_log_history(output_dir: str, log_history: List[Dict]):
    json_path = os.path.join(output_dir, "log_history.json")
    csv_path = os.path.join(output_dir, "log_history.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)

    keys = []
    for row in log_history:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in log_history:
            writer.writerow(row)

    print(f"Log history saved to: {json_path}")
    print(f"Log history saved to: {csv_path}")


def save_epoch_summary(output_dir: str, log_history: List[Dict]):
    """
    Creates a compact epoch-level CSV. It takes the last logged training loss
    within each epoch and the eval loss recorded at that epoch, if available.
    """
    epoch_rows: Dict[int, Dict] = {}

    for row in log_history:
        if "epoch" not in row:
            continue

        epoch_float = row["epoch"]
        if epoch_float is None:
            continue

        epoch_idx = max(1, int(math.ceil(float(epoch_float))))
        epoch_rows.setdefault(epoch_idx, {"epoch": epoch_idx})

        if "loss" in row:
            epoch_rows[epoch_idx]["train_loss_last_logged"] = row["loss"]
        if "eval_loss" in row:
            epoch_rows[epoch_idx]["eval_loss"] = row["eval_loss"]

    if not epoch_rows:
        return

    path = os.path.join(output_dir, "epoch_metrics_summary.csv")
    fieldnames = ["epoch", "train_loss_last_logged", "eval_loss"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch_idx in sorted(epoch_rows):
            writer.writerow(epoch_rows[epoch_idx])

    print(f"Epoch metrics summary saved to: {path}")


def plot_loss_curves(output_dir: str, log_history: List[Dict]):
    train_points = [
        (float(row["epoch"]), float(row["loss"]))
        for row in log_history
        if row.get("epoch") is not None and row.get("loss") is not None
    ]
    eval_points = [
        (float(row["epoch"]), float(row["eval_loss"]))
        for row in log_history
        if row.get("epoch") is not None and row.get("eval_loss") is not None
    ]

    if not train_points and not eval_points:
        print("No loss values found in trainer log history; skipping plot.")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Could not import matplotlib; skipping plot. Error: {e}")
        return

    plt.figure(figsize=(8, 5))

    if train_points:
        x_train, y_train = zip(*train_points)
        plt.plot(x_train, y_train, marker="o", label="Training loss")

    if eval_points:
        x_eval, y_eval = zip(*eval_points)
        plt.plot(x_eval, y_eval, marker="o", label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SFT LoRA training curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Loss curve saved to: {path}")


def build_training_args(args, has_eval_dataset: bool) -> TrainingArguments:
    """
    Keeps the script compatible with both newer Transformers versions that use
    eval_strategy and older versions that still use evaluation_strategy.
    """
    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        bf16=True,
        report_to=args.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
        ddp_find_unused_parameters=False,
    )

    signature_params = inspect.signature(TrainingArguments.__init__).parameters

    if has_eval_dataset:
        if "eval_strategy" in signature_params:
            kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in signature_params:
            kwargs["evaluation_strategy"] = "epoch"
        else:
            print(
                "Warning: this Transformers version does not expose eval_strategy/"
                "evaluation_strategy in TrainingArguments. Validation loss may not run."
            )
    else:
        if "eval_strategy" in signature_params:
            kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in signature_params:
            kwargs["evaluation_strategy"] = "no"

    # Some older versions may not support newer arguments. Drop unsupported keys.
    kwargs = {k: v for k, v in kwargs.items() if k in signature_params}
    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.0,
        help="Optional validation split from train_file, e.g. 0.1. Ignored if --eval_file is given.",
    )

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--report_to", type=str, default="none")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)

    # Keep local_files_only=True for cluster/offline use.
    parser.add_argument("--local_files_only", action="store_true", default=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("SFT LoRA training")
    print("Model:", args.model_id)
    print("Train file:", args.train_file)
    print("Eval file:", args.eval_file)
    print("Eval ratio:", args.eval_ratio)
    print("Output dir:", args.output_dir)
    print("Epochs:", args.epochs)
    print("Checkpointing: save_strategy='epoch' -> one checkpoint per epoch")
    print("=" * 80)

    rows = read_jsonl(args.train_file)
    num_total_examples = len(rows)

    if args.eval_file:
        train_rows = rows
        eval_rows = read_jsonl(args.eval_file)
    else:
        train_rows, eval_rows = split_train_eval(
            rows=rows,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
        )

    train_type_counts = count_types(train_rows)
    eval_type_counts = count_types(eval_rows) if eval_rows else None

    print(f"Loaded {num_total_examples} examples from train_file")
    print(f"Training examples: {len(train_rows)}")
    print("Training example types:", train_type_counts)

    if eval_rows:
        print(f"Validation examples: {len(eval_rows)}")
        print("Validation example types:", eval_type_counts)
    else:
        print("Validation examples: 0")
        print("Validation loss will not be computed. Use --eval_file or --eval_ratio for eval_loss.")

    schedule_info = compute_schedule_info(
        num_train_examples=len(train_rows),
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
    )

    print("World size:", schedule_info["world_size"])
    print("Effective batch size:", schedule_info["effective_batch_size"])
    print("Steps per epoch:", schedule_info["steps_per_epoch"])
    print("Total optimization steps:", schedule_info["total_optimization_steps"])

    write_run_metadata(
        output_dir=args.output_dir,
        args=args,
        num_total_examples=num_total_examples,
        train_type_counts=train_type_counts,
        eval_type_counts=eval_type_counts,
        schedule_info=schedule_info,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = MixedSFTDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        system_prompt=args.system_prompt,
    )

    eval_dataset = None
    if eval_rows:
        eval_dataset = MixedSFTDataset(
            rows=eval_rows,
            tokenizer=tokenizer,
            max_length=args.max_length,
            system_prompt=args.system_prompt,
        )

    training_args = build_training_args(args, has_eval_dataset=eval_dataset is not None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalLM(tokenizer),
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)

    save_log_history(args.output_dir, trainer.state.log_history)
    save_epoch_summary(args.output_dir, trainer.state.log_history)
    plot_loss_curves(args.output_dir, trainer.state.log_history)

    final_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)

    print("Saving final LoRA adapter...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("=" * 80)
    print("Training finished.")
    print("Intermediate epoch checkpoints saved under:", args.output_dir)
    print("Final adapter saved to:", final_dir)
    print("Training logs saved to:", os.path.join(args.output_dir, "log_history.json"))
    print("Training logs CSV saved to:", os.path.join(args.output_dir, "log_history.csv"))
    print("Epoch summary saved to:", os.path.join(args.output_dir, "epoch_metrics_summary.csv"))
    print("Loss curve saved to:", os.path.join(args.output_dir, "loss_curve.png"))
    print("=" * 80)


if __name__ == "__main__":
    main()
