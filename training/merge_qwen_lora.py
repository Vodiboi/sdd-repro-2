#!/usr/bin/env python3
"""Merge or export a trained Unsloth LoRA adapter."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_ADAPTER_DIR = Path("training_outputs/qwen3_4b_lora")
DEFAULT_OUTPUT_DIR = Path("training_outputs/qwen3_4b_merged_16bit")


def merge(args: argparse.Namespace) -> None:
    from unsloth import FastLanguageModel

    if args.hf_token_env and os.environ.get(args.hf_token_env):
        os.environ.setdefault("HF_TOKEN", os.environ[args.hf_token_env])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter_dir),
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        token=os.environ.get("HF_TOKEN"),
    )

    if args.export == "merged_16bit":
        model.save_pretrained_merged(
            str(args.output_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    elif args.export == "merged_4bit":
        model.save_pretrained_merged(
            str(args.output_dir),
            tokenizer,
            save_method="merged_4bit",
        )
    else:
        model.save_pretrained_gguf(
            str(args.output_dir),
            tokenizer,
            quantization_method=args.gguf_quantization,
        )

    print(f"Saved {args.export} artifact to {args.output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--push-to-hub requires --hub-model-id")
        token = os.environ.get("HF_TOKEN")
        if args.export == "gguf":
            model.push_to_hub_gguf(
                args.hub_model_id,
                tokenizer,
                quantization_method=args.gguf_quantization,
                token=token,
            )
        else:
            model.push_to_hub_merged(
                args.hub_model_id,
                tokenizer,
                save_method=args.export,
                token=token,
            )
        print(f"Pushed {args.export} artifact to {args.hub_model_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Directory produced by train_qwen_lora.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--export",
        choices=("merged_16bit", "merged_4bit", "gguf"),
        default="merged_16bit",
    )
    parser.add_argument("--gguf-quantization", default="q4_k_m")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser.parse_args()


def main() -> int:
    try:
        merge(parse_args())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
