#!/usr/bin/env python3
"""Build code-only SFT records from generated chat records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from train_qwen_lora import DEFAULT_DATA_PATH, jsonl_files_from_path, normalize_messages


DEFAULT_OUTPUT_DIR = Path("training/sft_code_only")
CODE_ONLY_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Return only a complete Python 3 "
    "solution. Do not include explanations, reasoning, markdown fences, or any "
    "text outside the code."
)


def strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_code_block(text: str) -> str | None:
    matches = list(
        re.finditer(
            r"```(?:python|py|Python)?[ \t]*\n(.*?)```",
            text,
            flags=re.DOTALL,
        )
    )
    if not matches:
        return None
    return matches[-1].group(1).strip()


def looks_like_prose(text: str) -> bool:
    lowered = text.lower()
    prose_markers = [
        "### explanation",
        "explanation:",
        "the code",
        "this solution",
        "we need",
        "let's",
    ]
    return any(marker in lowered for marker in prose_markers)


def clean_assistant_content(text: str) -> tuple[str, str]:
    without_thinking = strip_thinking_blocks(text)
    code = extract_code_block(without_thinking)
    if code is not None:
        return code, "code_block"
    return without_thinking.strip(), "stripped_text"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def convert(args: argparse.Namespace) -> dict[str, Any]:
    files = jsonl_files_from_path(args.data_path, prefer_sft=not args.prefer_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0
    source_counts: dict[str, int] = {}
    warnings: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    chunk_index = 1

    def flush_chunk() -> None:
        nonlocal output_rows, chunk_index
        if not output_rows:
            return
        write_jsonl(
            args.output_dir / "sft_messages" / f"sft_messages_{chunk_index:06d}.jsonl",
            output_rows,
        )
        output_rows = []
        chunk_index += 1

    for file_path in files:
        for line_number, record in enumerate((json.loads(line) for line in file_path.open(encoding="utf-8") if line.strip()), start=1):
            total += 1
            messages = normalize_messages(record, file_path, line_number)
            assistant = messages[-1]["content"]
            code, source = clean_assistant_content(assistant)
            source_counts[source] = source_counts.get(source, 0) + 1

            warning_reasons: list[str] = []
            if not code:
                warning_reasons.append("empty_after_cleaning")
            if "<think" in code.lower() or "</think" in code.lower():
                warning_reasons.append("thinking_marker_remaining")
            if "```" in code:
                warning_reasons.append("markdown_fence_remaining")
            if looks_like_prose(code):
                warning_reasons.append("possible_prose_remaining")

            if warning_reasons:
                warnings.append(
                    {
                        "source_file": str(file_path),
                        "line": line_number,
                        "reasons": warning_reasons,
                        "preview": code[:240],
                    }
                )
                if args.strict:
                    skipped += 1
                    continue

            cleaned_messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": messages[1]["content"]},
                {"role": "assistant", "content": code},
            ]
            output_rows.append({"messages": cleaned_messages})
            written += 1
            if len(output_rows) >= args.chunk_size:
                flush_chunk()

    flush_chunk()

    manifest = {
        "input_files": [str(path) for path in files],
        "output_dir": str(args.output_dir),
        "total_records": total,
        "written_records": written,
        "skipped_records": skipped,
        "assistant_source_counts": source_counts,
        "warning_count": len(warnings),
        "warnings_file": str(args.output_dir / "cleaning_warnings.jsonl"),
        "system_prompt": args.system_prompt,
        "strict": args.strict,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "cleaning_warnings.jsonl", warnings)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prefer-records",
        action="store_true",
        help="When --data-path is a directory, prefer full records files over SFT-only files.",
    )
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--system-prompt", default=CODE_ONLY_SYSTEM_PROMPT)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip records that still look like prose after cleaning.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        print(json.dumps(convert(parse_args()), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
