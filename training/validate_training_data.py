#!/usr/bin/env python3
"""Validate generated SSD records before starting GPU training."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import re
from pathlib import Path
from typing import Any

from train_qwen_lora import DEFAULT_DATA_PATH, jsonl_files_from_path, normalize_messages


def validate(path: Path, prefer_records: bool) -> dict[str, Any]:
    files = jsonl_files_from_path(path, prefer_sft=not prefer_records)
    role_counts: dict[str, int] = {}
    user_lengths: list[int] = []
    assistant_lengths: list[int] = []
    thinking_markers = 0
    markdown_fences = 0
    possible_prose = 0
    count = 0

    for file_path in files:
        with file_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                messages = normalize_messages(record, file_path, line_number)
                count += 1
                for message in messages:
                    role_counts[message["role"]] = role_counts.get(message["role"], 0) + 1
                user_lengths.append(len(messages[-2]["content"]))
                assistant = messages[-1]["content"]
                assistant_lengths.append(len(assistant))
                if "<think" in assistant.lower() or "</think" in assistant.lower():
                    thinking_markers += 1
                if "```" in assistant:
                    markdown_fences += 1
                if looks_like_prose(assistant):
                    possible_prose += 1

    if count == 0:
        raise ValueError("No records found")

    return {
        "records": count,
        "files": [str(file_path) for file_path in files],
        "role_counts": role_counts,
        "user_characters": length_summary(user_lengths),
        "total_user_assistant_characters": length_summary(
            [user + assistant for user, assistant in zip(user_lengths, assistant_lengths)]
        ),
        "assistant_characters": length_summary(assistant_lengths),
        "assistant_quality": {
            "thinking_marker_records": thinking_markers,
            "markdown_fence_records": markdown_fences,
            "possible_prose_records": possible_prose,
        },
    }


def length_summary(lengths: list[int]) -> dict[str, float | int]:
    return {
        "min": min(lengths),
        "p50": statistics.median(lengths),
        "p90": percentile(lengths, 0.90),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
    }


def percentile(values: list[int], q: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def looks_like_prose(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.lower())
    prose_markers = [
        "### explanation",
        "explanation:",
        "the code",
        "this solution",
        "we need",
        "let's",
    ]
    return any(marker in lowered for marker in prose_markers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--prefer-records",
        action="store_true",
        help="When --data-path is a directory, prefer full records files over SFT-only files.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        print(json.dumps(validate(args.data_path, args.prefer_records), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
