#!/usr/bin/env python3
"""Generate a Qwen3-4B simple self-distillation dataset from data 2.csv."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data 2.csv")
DEFAULT_OUTPUT_DIR = Path("ssd_qwen3_4b_dataset")
DEFAULT_MODEL_PATH = Path("/Users/aayanarish/models/Qwen3-4B-4bit")
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Solve the problem in Python "
    "and provide the final answer as a markdown code block."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_statement(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_dump_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def is_single_line_stub(text: str) -> bool:
    nonblank_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(nonblank_lines) <= 1


@dataclass(frozen=True)
class Candidate:
    id: str
    source_row_index: int
    contest: str
    problem_name: str
    problem_statement: str
    problem_tags: str
    normalized_statement: str
    statement_sha256: str
    duplicate_source_row_indices: list[int]


def read_candidates(input_path: Path) -> tuple[list[Candidate], list[dict[str, Any]], int]:
    candidates: list[Candidate] = []
    skipped: list[dict[str, Any]] = []
    seen: dict[str, Candidate] = {}
    total_rows = 0

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"contest", "problem_name", "problem_statement", "problem_tags"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{input_path} is missing required columns: {missing}")

        for zero_index, row in enumerate(reader):
            total_rows += 1
            source_row_index = zero_index + 2  # includes CSV header row
            statement = row.get("problem_statement", "")
            normalized = normalize_statement(statement)
            if not normalized:
                skipped.append(
                    {
                        "_skip_key": f"empty:{source_row_index}",
                        "reason": "empty_problem_statement",
                        "source_row_index": source_row_index,
                        "source": {
                            "contest": row.get("contest", ""),
                            "problem_name": row.get("problem_name", ""),
                            "problem_tags": row.get("problem_tags", ""),
                        },
                    }
                )
                continue

            statement_hash = sha256_text(normalized)
            if statement_hash in seen:
                original = seen[statement_hash]
                original.duplicate_source_row_indices.append(source_row_index)
                skipped.append(
                    {
                        "_skip_key": f"duplicate:{source_row_index}:{original.id}",
                        "reason": "duplicate_problem_statement",
                        "source_row_index": source_row_index,
                        "duplicate_of_id": original.id,
                        "duplicate_of_source_row_index": original.source_row_index,
                        "source": {
                            "contest": row.get("contest", ""),
                            "problem_name": row.get("problem_name", ""),
                            "problem_tags": row.get("problem_tags", ""),
                            "problem_statement_sha256": statement_hash,
                        },
                    }
                )
                continue

            base_id = (
                f"cf_{row.get('contest', '').strip() or 'unknown'}_"
                f"{row.get('problem_name', '').strip() or 'unknown'}_"
                f"{statement_hash[:12]}"
            )
            base_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_id)
            candidate = Candidate(
                id=base_id,
                source_row_index=source_row_index,
                contest=row.get("contest", ""),
                problem_name=row.get("problem_name", ""),
                problem_statement=statement,
                problem_tags=row.get("problem_tags", ""),
                normalized_statement=normalized,
                statement_sha256=statement_hash,
                duplicate_source_row_indices=[],
            )
            seen[statement_hash] = candidate
            candidates.append(candidate)

    return candidates, skipped, total_rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc


def read_completed_ids(records_path: Path) -> set[str]:
    ids: set[str] = set()
    for record in iter_jsonl(records_path):
        record_id = record.get("id")
        if record_id:
            ids.add(record_id)
    return ids


def read_skip_keys(skipped_path: Path) -> tuple[set[str], set[str], dict[str, int]]:
    skip_keys: set[str] = set()
    finalized_ids: set[str] = set()
    reason_counts: dict[str, int] = {}
    for record in iter_jsonl(skipped_path):
        key = record.get("_skip_key")
        if key:
            skip_keys.add(key)
        reason = record.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason in {"filtered_empty_output", "filtered_single_line_stub", "context_budget_exhausted"}:
            record_id = record.get("id")
            if record_id:
                finalized_ids.add(record_id)
    return skip_keys, finalized_ids, reason_counts


def build_messages(system_prompt: str, problem_statement: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_statement},
    ]


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_len(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def write_static_skips(
    skipped_path: Path,
    static_skips: list[dict[str, Any]],
    existing_skip_keys: set[str],
) -> int:
    written = 0
    with skipped_path.open("a", encoding="utf-8") as handle:
        for skip in static_skips:
            key = skip["_skip_key"]
            if key in existing_skip_keys:
                continue
            skip = dict(skip)
            skip["created_at"] = utc_now()
            json_dump_line(handle, skip)
            existing_skip_keys.add(key)
            written += 1
    return written


def make_manifest(
    *,
    args: argparse.Namespace,
    total_rows: int,
    candidates: list[Candidate],
    completed_ids: set[str],
    finalized_skipped_ids: set[str],
    skipped_reason_counts: dict[str, int],
    failures_this_run: int,
    filtered_this_run: int,
    generated_this_run: int,
    started_at: str,
) -> dict[str, Any]:
    static_empty = skipped_reason_counts.get("empty_problem_statement", 0)
    static_duplicates = skipped_reason_counts.get("duplicate_problem_statement", 0)
    accepted = len(completed_ids)
    finalized_without_record = len(finalized_skipped_ids - completed_ids)
    pending = max(0, len(candidates) - accepted - finalized_without_record)
    return {
        "created_or_updated_at": utc_now(),
        "run_started_at": started_at,
        "input_csv": str(args.input),
        "output_dir": str(args.output_dir),
        "model_path": str(args.model_path),
        "counts": {
            "input_rows": total_rows,
            "unique_nonempty_candidates": len(candidates),
            "accepted_records": accepted,
            "pending_unique_candidates": pending,
            "skipped_empty_problem_statement": static_empty,
            "skipped_duplicate_problem_statement": static_duplicates,
            "finalized_filtered_or_context_skips": finalized_without_record,
            "failures_logged": skipped_reason_counts.get("generation_exception", 0),
            "generated_this_run": generated_this_run,
            "filtered_this_run": filtered_this_run,
            "failures_this_run": failures_this_run,
        },
        "settings": {
            "samples_per_prompt": 1,
            "max_context_tokens": args.max_context_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "system_prompt": args.system_prompt,
            "paper_minimal_filtering": True,
            "no_correctness_verification": True,
            "no_code_execution": True,
            "seed": args.seed,
        },
        "files": {
            "records": "records.jsonl",
            "sft_messages": "sft_messages.jsonl",
            "skipped_rows": "skipped_rows.jsonl",
            "progress_state": "progress_state.json",
        },
    }


def validate_outputs(output_dir: Path, total_rows: int | None = None) -> dict[str, Any]:
    records_path = output_dir / "records.jsonl"
    sft_path = output_dir / "sft_messages.jsonl"
    skipped_path = output_dir / "skipped_rows.jsonl"

    record_ids: set[str] = set()
    max_total_tokens = 0
    record_count = 0
    for record in iter_jsonl(records_path):
        record_count += 1
        record_id = record.get("id")
        if not record_id:
            raise ValueError(f"{records_path} has a record without id")
        if record_id in record_ids:
            raise ValueError(f"{records_path} has duplicate id {record_id}")
        record_ids.add(record_id)
        messages = record.get("messages", [])
        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{record_id} messages roles are {roles}, expected system/user/assistant")
        total_tokens = record.get("token_counts", {}).get("total_tokens", 0)
        max_total_tokens = max(max_total_tokens, total_tokens)
        if total_tokens > record.get("generation_settings", {}).get("max_context_tokens", 32768):
            raise ValueError(f"{record_id} exceeds context window: {total_tokens}")

    sft_count = 0
    for record in iter_jsonl(sft_path):
        sft_count += 1
        messages = record.get("messages", [])
        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{sft_path} line {sft_count} roles are {roles}")

    if sft_count != record_count:
        raise ValueError(f"record count {record_count} != SFT count {sft_count}")

    skip_count = sum(1 for _ in iter_jsonl(skipped_path))
    reconciled = None
    if total_rows is not None:
        reconciled = record_count + skip_count == total_rows

    return {
        "records": record_count,
        "sft_messages": sft_count,
        "skipped_rows": skip_count,
        "max_total_tokens": max_total_tokens,
        "reconciles_to_input_rows": reconciled,
    }


def run_generation(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    sft_path = args.output_dir / "sft_messages.jsonl"
    skipped_path = args.output_dir / "skipped_rows.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    progress_path = args.output_dir / "progress_state.json"

    for jsonl_path in (records_path, sft_path, skipped_path):
        jsonl_path.touch(exist_ok=True)

    candidates, static_skips, total_rows = read_candidates(args.input)
    completed_ids = read_completed_ids(records_path)
    existing_skip_keys, finalized_skipped_ids, skipped_reason_counts = read_skip_keys(skipped_path)
    write_static_skips(skipped_path, static_skips, existing_skip_keys)
    existing_skip_keys, finalized_skipped_ids, skipped_reason_counts = read_skip_keys(skipped_path)

    started_at = utc_now()
    generated_this_run = 0
    filtered_this_run = 0
    failures_this_run = 0

    if args.validate_only:
        print(json.dumps(validate_outputs(args.output_dir, total_rows), indent=2, sort_keys=True))
        return

    if args.prepare_only:
        manifest = make_manifest(
            args=args,
            total_rows=total_rows,
            candidates=candidates,
            completed_ids=completed_ids,
            finalized_skipped_ids=finalized_skipped_ids,
            skipped_reason_counts=skipped_reason_counts,
            failures_this_run=failures_this_run,
            filtered_this_run=filtered_this_run,
            generated_this_run=generated_this_run,
            started_at=started_at,
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(progress_path, {"updated_at": utc_now(), "status": "prepared"})
        print(f"Prepared {args.output_dir} without loading the model.")
        return

    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    if args.seed is not None:
        mx.random.seed(args.seed)

    model, tokenizer = load(str(args.model_path))
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)

    with records_path.open("a", encoding="utf-8") as records_handle, sft_path.open(
        "a", encoding="utf-8"
    ) as sft_handle, skipped_path.open("a", encoding="utf-8") as skipped_handle:
        for candidate in candidates:
            if args.stop_after is not None and generated_this_run >= args.stop_after:
                break
            if candidate.id in completed_ids or candidate.id in finalized_skipped_ids:
                continue

            messages = build_messages(args.system_prompt, candidate.problem_statement)
            prompt = apply_chat_template(tokenizer, messages)
            prompt_tokens = encode_len(tokenizer, prompt)
            max_new_tokens = args.max_context_tokens - prompt_tokens

            if max_new_tokens <= 0:
                skip = {
                    "_skip_key": f"context:{candidate.id}",
                    "id": candidate.id,
                    "reason": "context_budget_exhausted",
                    "source_row_index": candidate.source_row_index,
                    "prompt_tokens": prompt_tokens,
                    "max_context_tokens": args.max_context_tokens,
                    "created_at": utc_now(),
                }
                json_dump_line(skipped_handle, skip)
                existing_skip_keys.add(skip["_skip_key"])
                finalized_skipped_ids.add(candidate.id)
                filtered_this_run += 1
                continue

            print(
                f"[{generated_this_run + 1}] generating {candidate.id} "
                f"(prompt_tokens={prompt_tokens}, max_new_tokens={max_new_tokens})",
                flush=True,
            )

            start = time.time()
            output_text = ""
            generation_tokens = 0
            finish_reason = None
            try:
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max_new_tokens,
                    sampler=sampler,
                ):
                    output_text += response.text
                    generation_tokens = response.generation_tokens
                    finish_reason = response.finish_reason
            except Exception as exc:  # Keep the long run moving.
                failures_this_run += 1
                skip = {
                    "_skip_key": f"generation_exception:{candidate.id}:{int(time.time())}",
                    "id": candidate.id,
                    "reason": "generation_exception",
                    "source_row_index": candidate.source_row_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "created_at": utc_now(),
                }
                json_dump_line(skipped_handle, skip)
                continue

            output_text = output_text.strip()
            output_tokens = encode_len(tokenizer, output_text)
            total_tokens = prompt_tokens + output_tokens

            if not output_text or is_single_line_stub(output_text):
                reason = "filtered_empty_output" if not output_text else "filtered_single_line_stub"
                skip = {
                    "_skip_key": f"{reason}:{candidate.id}",
                    "id": candidate.id,
                    "reason": reason,
                    "source_row_index": candidate.source_row_index,
                    "source": {
                        "contest": candidate.contest,
                        "problem_name": candidate.problem_name,
                        "problem_tags": candidate.problem_tags,
                        "problem_statement_sha256": candidate.statement_sha256,
                    },
                    "token_counts": {
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    },
                    "created_at": utc_now(),
                }
                json_dump_line(skipped_handle, skip)
                existing_skip_keys.add(skip["_skip_key"])
                finalized_skipped_ids.add(candidate.id)
                filtered_this_run += 1
                continue

            full_messages = messages + [{"role": "assistant", "content": output_text}]
            generation_settings = {
                "model_path": str(args.model_path),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "max_context_tokens": args.max_context_tokens,
                "max_new_tokens": max_new_tokens,
                "samples_per_prompt": 1,
            }
            record = {
                "id": candidate.id,
                "created_at": utc_now(),
                "source": {
                    "csv": str(args.input),
                    "row_index": candidate.source_row_index,
                    "contest": candidate.contest,
                    "problem_name": candidate.problem_name,
                    "problem_tags": candidate.problem_tags,
                    "problem_statement_sha256": candidate.statement_sha256,
                    "duplicate_source_row_indices": candidate.duplicate_source_row_indices,
                },
                "messages": full_messages,
                "token_counts": {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "generation_tokens_reported_by_mlx": generation_tokens,
                },
                "generation_settings": generation_settings,
                "finish_reason": finish_reason,
                "elapsed_seconds": round(time.time() - start, 3),
            }
            json_dump_line(records_handle, record)
            json_dump_line(sft_handle, {"messages": full_messages})
            completed_ids.add(candidate.id)
            generated_this_run += 1

            if generated_this_run % args.checkpoint_every == 0:
                existing_skip_keys, finalized_skipped_ids, skipped_reason_counts = read_skip_keys(skipped_path)
                manifest = make_manifest(
                    args=args,
                    total_rows=total_rows,
                    candidates=candidates,
                    completed_ids=completed_ids,
                    finalized_skipped_ids=finalized_skipped_ids,
                    skipped_reason_counts=skipped_reason_counts,
                    failures_this_run=failures_this_run,
                    filtered_this_run=filtered_this_run,
                    generated_this_run=generated_this_run,
                    started_at=started_at,
                )
                atomic_write_json(manifest_path, manifest)
                atomic_write_json(
                    progress_path,
                    {
                        "updated_at": utc_now(),
                        "status": "running",
                        "last_completed_id": candidate.id,
                        "generated_this_run": generated_this_run,
                    },
                )

    existing_skip_keys, finalized_skipped_ids, skipped_reason_counts = read_skip_keys(skipped_path)
    manifest = make_manifest(
        args=args,
        total_rows=total_rows,
        candidates=candidates,
        completed_ids=completed_ids,
        finalized_skipped_ids=finalized_skipped_ids,
        skipped_reason_counts=skipped_reason_counts,
        failures_this_run=failures_this_run,
        filtered_this_run=filtered_this_run,
        generated_this_run=generated_this_run,
        started_at=started_at,
    )
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        progress_path,
        {
            "updated_at": utc_now(),
            "status": "complete_or_stopped",
            "generated_this_run": generated_this_run,
            "filtered_this_run": filtered_this_run,
            "failures_this_run": failures_this_run,
        },
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Stop after this many newly accepted records. Useful for dry runs.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        run_generation(parse_args())
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
