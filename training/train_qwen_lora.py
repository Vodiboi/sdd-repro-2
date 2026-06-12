#!/usr/bin/env python3
"""Fine-tune Qwen3 with Unsloth LoRA on generated SSD chat records."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
DEFAULT_DATA_PATH = Path("training/sft_messages")
DEFAULT_OUTPUT_DIR = Path("training_outputs/qwen3_4b_lora")
DEFAULT_CHAT_TEMPLATE = "qwen3-instruct"
RESPONSE_MARKER = "<|im_start|>assistant\n"
INSTRUCTION_MARKER = "<|im_start|>user\n"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc


def jsonl_files_from_path(path: Path, prefer_sft: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Training data path does not exist: {path}")

    direct_sft_chunks = sorted(path.glob("sft_messages_*.jsonl"))
    direct_record_chunks = sorted(path.glob("records_*.jsonl"))
    direct_jsonl_files = sorted(
        file_path
        for file_path in path.glob("*.jsonl")
        if file_path.name not in {"skipped_rows.jsonl"}
    )

    candidate_groups = []
    if prefer_sft:
        candidate_groups.extend(
            [
                sorted((path / "sft_messages").glob("sft_messages_*.jsonl")),
                direct_sft_chunks,
                [path / "sft_messages.jsonl"],
                sorted((path / "records").glob("records_*.jsonl")),
                direct_record_chunks,
                [path / "records.jsonl"],
                direct_jsonl_files,
            ]
        )
    else:
        candidate_groups.extend(
            [
                sorted((path / "records").glob("records_*.jsonl")),
                direct_record_chunks,
                [path / "records.jsonl"],
                sorted((path / "sft_messages").glob("sft_messages_*.jsonl")),
                direct_sft_chunks,
                [path / "sft_messages.jsonl"],
                direct_jsonl_files,
            ]
        )

    for group in candidate_groups:
        files = [file_path for file_path in group if file_path.exists()]
        if files:
            return files
    raise FileNotFoundError(
        f"No records JSONL files found under {path}. Expected records.jsonl, "
        "sft_messages.jsonl, records/*.jsonl, sft_messages/*.jsonl, or a "
        "directory of JSONL chunk files."
    )


def normalize_messages(record: dict[str, Any], source: Path, index: int) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{source} record {index} does not contain a messages list")

    normalized: list[dict[str, str]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{source} record {index} message {message_index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{source} record {index} has invalid role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{source} record {index} has empty content for role {role!r}")
        normalized.append({"role": role, "content": content})

    roles = [message["role"] for message in normalized]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"{source} record {index} roles are {roles}, expected system/user/assistant")
    return normalized


def load_chat_records(paths: list[Path], limit: int | None, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for index, record in enumerate(iter_jsonl(path), start=1):
            rows.append({"conversations": normalize_messages(record, path, index)})

    if not rows:
        raise ValueError("No training records were loaded")

    random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


def split_rows(
    rows: list[dict[str, Any]],
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    if eval_fraction <= 0:
        return rows, None
    if not 0 < eval_fraction < 1:
        raise ValueError("--eval-fraction must be between 0 and 1")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, int(round(len(shuffled) * eval_fraction)))
    if eval_size >= len(shuffled):
        raise ValueError("--eval-fraction leaves no training records")
    return shuffled[eval_size:], shuffled[:eval_size]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def add_lora(model: Any, args: argparse.Namespace) -> Any:
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.gradient_checkpointing,
        random_state=args.seed,
        use_rslora=args.use_rslora,
        loftq_config=None,
    )


def make_sft_config(SFTConfig: Any, args: argparse.Namespace) -> Any:
    parameters = inspect.signature(SFTConfig).parameters
    requested_kwargs = {
        "output_dir": str(args.output_dir),
        "dataset_text_field": "text",
        "packing": args.packing,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
        "report_to": args.report_to,
    }

    if "max_seq_length" in parameters:
        requested_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in parameters:
        requested_kwargs["max_length"] = args.max_seq_length

    config_kwargs = {
        key: value
        for key, value in requested_kwargs.items()
        if key in parameters
    }
    skipped = sorted(set(requested_kwargs) - set(config_kwargs))
    if skipped:
        print(f"Skipping unsupported SFTConfig args for this TRL version: {skipped}")
    return SFTConfig(**config_kwargs)


def make_sft_trainer(
    SFTTrainer: Any,
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
) -> Any:
    parameters = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": args,
    }
    if "tokenizer" in parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in parameters:
        trainer_kwargs["processing_class"] = tokenizer
    return SFTTrainer(**trainer_kwargs)


def train(args: argparse.Namespace) -> None:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if args.hf_token_env and os.environ.get(args.hf_token_env):
        os.environ.setdefault("HF_TOKEN", os.environ[args.hf_token_env])

    data_paths = jsonl_files_from_path(args.data_path, prefer_sft=not args.prefer_records)
    rows = load_chat_records(data_paths, args.limit_records, args.seed)
    train_rows, eval_rows = split_rows(rows, args.eval_fraction, args.seed)

    print(f"Using data files: {[str(path) for path in data_paths]}")
    print(f"Loaded {len(rows)} records; train={len(train_rows)}, eval={len(eval_rows or [])}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=False,
        token=os.environ.get("HF_TOKEN"),
    )
    model = add_lora(model, args)
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    train_dataset = standardize_data_formats(Dataset.from_list(train_rows))
    eval_dataset = standardize_data_formats(Dataset.from_list(eval_rows)) if eval_rows else None

    def formatting_prompts_func(examples: dict[str, Any]) -> dict[str, list[str]]:
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    training_args = make_sft_config(SFTConfig, args)

    trainer = make_sft_trainer(
        SFTTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_MARKER,
        response_part=RESPONSE_MARKER,
    )

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(trainer_stats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_adapter:
        model.save_pretrained(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        print(f"Saved LoRA adapter to {args.output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--push-to-hub requires --hub-model-id")
        model.push_to_hub(args.hub_model_id, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(args.hub_model_id, token=os.environ.get("HF_TOKEN"))
        print(f"Pushed LoRA adapter to {args.hub_model_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--prefer-records",
        action="store_true",
        help="When --data-path is a directory, prefer full records files over SFT-only files.",
    )
    parser.add_argument("--limit-records", type=positive_int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.0)

    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--chat-template", default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--max-seq-length", type=positive_int, default=2048)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--lora-rank", type=positive_int, default=32)
    parser.add_argument("--lora-alpha", type=positive_int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--gradient-checkpointing", default="unsloth")
    parser.add_argument("--use-rslora", action="store_true")

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-device-train-batch-size", type=positive_int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--logging-steps", type=positive_int, default=1)
    parser.add_argument("--save-steps", type=positive_int, default=25)
    parser.add_argument("--save-total-limit", type=positive_int, default=2)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", default=None)

    parser.add_argument("--save-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing a Hugging Face token.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        train(parse_args())
    except KeyboardInterrupt:
        print("Interrupted; rerun with --resume-from-checkpoint if a checkpoint was saved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
