#!/usr/bin/env python3
"""Generate a Qwen3-4B simple self-distillation dataset from LiveCodeBench."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_DATASET_NAME = "livecodebench/code_generation_lite"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_OUTPUT_DIR = Path("ssd_qwen3_4b_lcb_dataset")
DEFAULT_MLX_MODEL = "/Users/aayanarish/models/Qwen3-4B-4bit"
DEFAULT_CHUNK_SIZE = 100
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Return only a complete Python 3 "
    "solution. Do not include explanations, reasoning, markdown fences, or any "
    "text outside the code."
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
    source_index: int
    prompt_text: str
    normalized_prompt: str
    prompt_sha256: str
    source_metadata: dict[str, Any]
    duplicate_source_indices: list[int]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: str | None
    token_counts: dict[str, Any]


@dataclass(frozen=True)
class GenerationBackend:
    name: str
    model: str
    tokenizer: Any
    generate: Callable[[str, int], GenerationResult]


@dataclass(frozen=True)
class ChunkState:
    index: int
    record_count: int


def build_lcb_prompt(row: dict[str, Any]) -> str:
    question_content = row.get("question_content") or ""
    starter_code = (row.get("starter_code") or "").strip()
    if not starter_code:
        return question_content
    return (
        question_content.rstrip()
        + "\n\nStarter code:\n```python\n"
        + starter_code
        + "\n```"
    )


def livecodebench_metadata(row: dict[str, Any]) -> dict[str, Any]:
    public_test_cases = row.get("public_test_cases")
    private_test_cases = row.get("private_test_cases")
    return {
        "dataset": DEFAULT_DATASET_NAME,
        "question_title": row.get("question_title", ""),
        "question_id": row.get("question_id", ""),
        "platform": row.get("platform", ""),
        "contest_id": row.get("contest_id", ""),
        "contest_date": row.get("contest_date", ""),
        "difficulty": row.get("difficulty", ""),
        "metadata": row.get("metadata", ""),
        "has_starter_code": bool((row.get("starter_code") or "").strip()),
        "has_public_test_cases": bool(public_test_cases),
        "has_private_test_cases": bool(private_test_cases),
    }


def read_candidates(args: argparse.Namespace) -> tuple[list[Candidate], list[dict[str, Any]], int]:
    from datasets import load_dataset

    candidates: list[Candidate] = []
    skipped: list[dict[str, Any]] = []
    seen: dict[str, Candidate] = {}
    total_rows = 0

    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        trust_remote_code=args.trust_remote_code,
    )

    for zero_index, row in enumerate(dataset):
        total_rows += 1
        source_index = zero_index
        prompt_text = build_lcb_prompt(row)
        normalized = normalize_statement(prompt_text)
        metadata = livecodebench_metadata(row)
        metadata["dataset"] = args.dataset_name
        metadata["split"] = args.dataset_split

        if not normalized:
            skipped.append(
                {
                    "_skip_key": f"empty:{source_index}",
                    "reason": "empty_prompt",
                    "source_index": source_index,
                    "source": metadata,
                }
            )
            continue

        prompt_hash = sha256_text(normalized)
        if prompt_hash in seen:
            original = seen[prompt_hash]
            original.duplicate_source_indices.append(source_index)
            skipped.append(
                {
                    "_skip_key": f"duplicate:{source_index}:{original.id}",
                    "reason": "duplicate_prompt",
                    "source_index": source_index,
                    "duplicate_of_id": original.id,
                    "duplicate_of_source_index": original.source_index,
                    "source": {
                        **metadata,
                        "prompt_sha256": prompt_hash,
                    },
                }
            )
            continue

        question_id = str(row.get("question_id") or f"row_{source_index}")
        title = str(row.get("question_title") or "untitled")
        base_id = f"lcb_{question_id}_{title}_{prompt_hash[:12]}"
        base_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_id).strip("_")
        candidate = Candidate(
            id=base_id,
            source_index=source_index,
            prompt_text=prompt_text,
            normalized_prompt=normalized,
            prompt_sha256=prompt_hash,
            source_metadata=metadata,
            duplicate_source_indices=[],
        )
        seen[prompt_hash] = candidate
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


def chunk_path(output_dir: Path, directory_name: str, prefix: str, index: int) -> Path:
    return output_dir / directory_name / f"{prefix}_{index:06d}.jsonl"


def chunk_indices(output_dir: Path, directory_name: str, prefix: str) -> list[int]:
    directory = output_dir / directory_name
    if not directory.exists():
        return []

    indices: list[int] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.jsonl$")
    for path in directory.glob(f"{prefix}_*.jsonl"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def output_jsonl_files(
    output_dir: Path,
    *,
    legacy_name: str,
    directory_name: str,
    prefix: str,
) -> list[Path]:
    files: list[Path] = []
    legacy_path = output_dir / legacy_name
    if legacy_path.exists():
        files.append(legacy_path)
    for index in chunk_indices(output_dir, directory_name, prefix):
        files.append(chunk_path(output_dir, directory_name, prefix, index))
    return files


def record_files(output_dir: Path) -> list[Path]:
    return output_jsonl_files(
        output_dir,
        legacy_name="records.jsonl",
        directory_name="records",
        prefix="records",
    )


def sft_files(output_dir: Path) -> list[Path]:
    return output_jsonl_files(
        output_dir,
        legacy_name="sft_messages.jsonl",
        directory_name="sft_messages",
        prefix="sft_messages",
    )


def count_jsonl_records(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def next_chunk_state(output_dir: Path, chunk_size: int) -> ChunkState:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer")

    indices = chunk_indices(output_dir, "records", "records")
    if not indices:
        return ChunkState(index=1, record_count=0)

    last_index = indices[-1]
    last_count = count_jsonl_records(chunk_path(output_dir, "records", "records", last_index))
    if last_count >= chunk_size:
        return ChunkState(index=last_index + 1, record_count=0)
    return ChunkState(index=last_index, record_count=last_count)


def write_accepted_chunk(
    output_dir: Path,
    chunk_index: int,
    record: dict[str, Any],
    sft_record: dict[str, Any],
) -> None:
    records_path = chunk_path(output_dir, "records", "records", chunk_index)
    sft_path = chunk_path(output_dir, "sft_messages", "sft_messages", chunk_index)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as records_handle:
        json_dump_line(records_handle, record)
    with sft_path.open("a", encoding="utf-8") as sft_handle:
        json_dump_line(sft_handle, sft_record)


def read_completed_ids(output_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in record_files(output_dir):
        for record in iter_jsonl(path):
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


def code_only_output(text: str) -> str:
    text = strip_thinking_blocks(text)
    code = extract_code_block(text)
    return code if code is not None else text.strip()


def build_messages(system_prompt: str, prompt_text: str, no_think_suffix: bool) -> list[dict[str, str]]:
    if no_think_suffix:
        prompt_text = prompt_text.rstrip() + "\n\n/no_think"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_len(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def resolve_backend(requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    if platform.system() == "Darwin":
        return "mlx"
    return "vllm"


def normalize_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    args.requested_backend = args.backend
    args.backend = resolve_backend(args.backend)

    if args.model is None:
        if args.backend == "mlx":
            args.model = DEFAULT_MLX_MODEL
        else:
            raise ValueError(
                "A vLLM run needs an explicit HF-format model path or model id. "
                "Pass --model /path/to/HF-format-Qwen3-4B."
            )

    if args.backend == "vllm" and str(args.model) == DEFAULT_MLX_MODEL:
        raise ValueError(
            f"{DEFAULT_MLX_MODEL} is an MLX-format model and cannot run on an A100. "
            "Pass --model pointing at a Hugging Face-format Qwen3-4B directory."
        )

    return args


def load_mlx_backend(args: argparse.Namespace) -> GenerationBackend:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    if args.seed is not None:
        mx.random.seed(args.seed)

    model, tokenizer = load(str(args.model))
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)

    def generate(prompt: str, max_new_tokens: int) -> GenerationResult:
        output_text = ""
        generation_tokens = 0
        finish_reason = None
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
        return GenerationResult(
            text=output_text,
            finish_reason=finish_reason,
            token_counts={"generation_tokens_reported_by_mlx": generation_tokens},
        )

    return GenerationBackend(
        name="mlx",
        model=str(args.model),
        tokenizer=tokenizer,
        generate=generate,
    )


def load_vllm_backend(args: argparse.Namespace) -> GenerationBackend:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model),
        trust_remote_code=args.model_trust_remote_code,
        dtype=args.vllm_dtype,
        max_model_len=args.max_context_tokens,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()

    def make_sampling_params(max_new_tokens: int) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": max_new_tokens,
        }
        if args.seed is not None:
            kwargs["seed"] = args.seed
        try:
            return SamplingParams(**kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            return SamplingParams(**kwargs)

    def generate(prompt: str, max_new_tokens: int) -> GenerationResult:
        outputs = llm.generate([prompt], make_sampling_params(max_new_tokens))
        if not outputs or not outputs[0].outputs:
            return GenerationResult(
                text="",
                finish_reason=None,
                token_counts={"generation_tokens_reported_by_vllm": 0},
            )
        completion = outputs[0].outputs[0]
        token_ids = getattr(completion, "token_ids", None)
        generation_tokens = len(token_ids) if token_ids is not None else None
        return GenerationResult(
            text=completion.text,
            finish_reason=getattr(completion, "finish_reason", None),
            token_counts={"generation_tokens_reported_by_vllm": generation_tokens},
        )

    return GenerationBackend(
        name="vllm",
        model=str(args.model),
        tokenizer=tokenizer,
        generate=generate,
    )


def load_generation_backend(args: argparse.Namespace) -> GenerationBackend:
    if args.backend == "mlx":
        return load_mlx_backend(args)
    if args.backend == "vllm":
        return load_vllm_backend(args)
    raise ValueError(f"Unsupported backend: {args.backend}")


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
    static_empty = skipped_reason_counts.get("empty_prompt", 0)
    static_duplicates = skipped_reason_counts.get("duplicate_prompt", 0)
    accepted = len(completed_ids)
    finalized_without_record = len(finalized_skipped_ids - completed_ids)
    pending = max(0, len(candidates) - accepted - finalized_without_record)
    return {
        "created_or_updated_at": utc_now(),
        "run_started_at": started_at,
        "source_dataset": args.dataset_name,
        "source_split": args.dataset_split,
        "output_dir": str(args.output_dir),
        "backend": args.backend,
        "requested_backend": args.requested_backend,
        "model": str(args.model),
        "counts": {
            "source_rows": total_rows,
            "unique_nonempty_prompts": len(candidates),
            "accepted_records": accepted,
            "pending_unique_prompts": pending,
            "skipped_empty_prompt": static_empty,
            "skipped_duplicate_prompt": static_duplicates,
            "finalized_filtered_or_context_skips": finalized_without_record,
            "failures_logged": skipped_reason_counts.get("generation_exception", 0),
            "generated_this_run": generated_this_run,
            "filtered_this_run": filtered_this_run,
            "failures_this_run": failures_this_run,
        },
        "settings": {
            "samples_per_prompt": 1,
            "chunk_size": args.chunk_size,
            "max_context_tokens": args.max_context_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "system_prompt": args.system_prompt,
            "enable_thinking": args.enable_thinking,
            "no_think_suffix": args.no_think_suffix,
            "code_only_sft": args.code_only_sft,
            "paper_minimal_filtering": True,
            "no_correctness_verification": True,
            "no_code_execution": True,
            "seed": args.seed,
            "model_trust_remote_code": args.model_trust_remote_code,
            "vllm_dtype": args.vllm_dtype,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        },
        "files": {
            "records": "records/records_*.jsonl",
            "sft_messages": "sft_messages/sft_messages_*.jsonl",
            "legacy_records": "records.jsonl",
            "legacy_sft_messages": "sft_messages.jsonl",
            "skipped_rows": "skipped_rows.jsonl",
            "progress_state": "progress_state.json",
        },
    }


def validate_outputs(output_dir: Path, total_rows: int | None = None) -> dict[str, Any]:
    skipped_path = output_dir / "skipped_rows.jsonl"
    records_paths = record_files(output_dir)
    sft_paths = sft_files(output_dir)

    record_ids: set[str] = set()
    max_total_tokens = 0
    record_count = 0
    for path in records_paths:
        for record in iter_jsonl(path):
            record_count += 1
            record_id = record.get("id")
            if not record_id:
                raise ValueError(f"{path} has a record without id")
            if record_id in record_ids:
                raise ValueError(f"{path} has duplicate id {record_id}")
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
    for path in sft_paths:
        for record in iter_jsonl(path):
            sft_count += 1
            messages = record.get("messages", [])
            roles = [message.get("role") for message in messages]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"{path} line {sft_count} roles are {roles}")

    if sft_count != record_count:
        raise ValueError(f"record count {record_count} != SFT count {sft_count}")

    skip_count = sum(1 for _ in iter_jsonl(skipped_path))
    reconciled = None
    if total_rows is not None:
        reconciled = record_count + skip_count == total_rows

    return {
        "records": record_count,
        "record_files": [str(path) for path in records_paths],
        "sft_messages": sft_count,
        "sft_message_files": [str(path) for path in sft_paths],
        "skipped_rows": skip_count,
        "max_total_tokens": max_total_tokens,
        "reconciles_to_source_rows": reconciled,
    }


def run_generation(args: argparse.Namespace) -> None:
    args = normalize_runtime_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records").mkdir(exist_ok=True)
    (args.output_dir / "sft_messages").mkdir(exist_ok=True)
    skipped_path = args.output_dir / "skipped_rows.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    progress_path = args.output_dir / "progress_state.json"

    skipped_path.touch(exist_ok=True)

    candidates, static_skips, total_rows = read_candidates(args)
    completed_ids = read_completed_ids(args.output_dir)
    existing_skip_keys, finalized_skipped_ids, skipped_reason_counts = read_skip_keys(skipped_path)
    write_static_skips(skipped_path, static_skips, existing_skip_keys)
    existing_skip_keys, finalized_skipped_ids, skipped_reason_counts = read_skip_keys(skipped_path)
    chunk_state = next_chunk_state(args.output_dir, args.chunk_size)
    current_chunk_index = chunk_state.index
    current_chunk_record_count = chunk_state.record_count

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
        atomic_write_json(
            progress_path,
            {
                "updated_at": utc_now(),
                "status": "prepared",
                "accepted_records": len(completed_ids),
                "current_chunk_index": current_chunk_index,
                "current_chunk_record_count": current_chunk_record_count,
            },
        )
        print(f"Prepared {args.output_dir} without loading the model.")
        return

    backend = load_generation_backend(args)
    tokenizer = backend.tokenizer

    with skipped_path.open("a", encoding="utf-8") as skipped_handle:
        for candidate in candidates:
            if args.stop_after is not None and generated_this_run >= args.stop_after:
                break
            if candidate.id in completed_ids or candidate.id in finalized_skipped_ids:
                continue

            messages = build_messages(args.system_prompt, candidate.prompt_text, args.no_think_suffix)
            prompt = apply_chat_template(tokenizer, messages, args.enable_thinking)
            prompt_tokens = encode_len(tokenizer, prompt)
            max_new_tokens = args.max_context_tokens - prompt_tokens

            if max_new_tokens <= 0:
                skip = {
                    "_skip_key": f"context:{candidate.id}",
                    "id": candidate.id,
                    "reason": "context_budget_exhausted",
                    "source_index": candidate.source_index,
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
            try:
                generation = backend.generate(prompt, max_new_tokens)
            except Exception as exc:  # Keep the long run moving.
                failures_this_run += 1
                skip = {
                    "_skip_key": f"generation_exception:{candidate.id}:{int(time.time())}",
                    "id": candidate.id,
                    "reason": "generation_exception",
                    "source_index": candidate.source_index,
                    "source": candidate.source_metadata,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "created_at": utc_now(),
                }
                json_dump_line(skipped_handle, skip)
                continue

            raw_output_text = generation.text.strip()
            output_text = code_only_output(raw_output_text) if args.code_only_sft else raw_output_text
            output_tokens = encode_len(tokenizer, output_text)
            total_tokens = prompt_tokens + output_tokens

            if not output_text or is_single_line_stub(output_text):
                reason = "filtered_empty_output" if not output_text else "filtered_single_line_stub"
                skip = {
                    "_skip_key": f"{reason}:{candidate.id}",
                    "id": candidate.id,
                    "reason": reason,
                    "source_index": candidate.source_index,
                    "source": {
                        **candidate.source_metadata,
                        "prompt_sha256": candidate.prompt_sha256,
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
                "backend": backend.name,
                "model": backend.model,
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
                    **candidate.source_metadata,
                    "source_index": candidate.source_index,
                    "prompt_sha256": candidate.prompt_sha256,
                    "duplicate_source_indices": candidate.duplicate_source_indices,
                },
                "messages": full_messages,
                "token_counts": {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    **generation.token_counts,
                },
                "generation_settings": generation_settings,
                "finish_reason": generation.finish_reason,
                "elapsed_seconds": round(time.time() - start, 3),
            }
            if raw_output_text != output_text:
                record["raw_assistant_content"] = raw_output_text
            write_accepted_chunk(
                args.output_dir,
                current_chunk_index,
                record,
                {"messages": full_messages},
            )
            completed_ids.add(candidate.id)
            generated_this_run += 1
            current_chunk_record_count += 1
            if current_chunk_record_count >= args.chunk_size:
                current_chunk_index += 1
                current_chunk_record_count = 0

            atomic_write_json(
                progress_path,
                {
                    "updated_at": utc_now(),
                    "status": "running",
                    "last_completed_id": candidate.id,
                    "accepted_records": len(completed_ids),
                    "generated_this_run": generated_this_run,
                    "current_chunk_index": current_chunk_index,
                    "current_chunk_record_count": current_chunk_record_count,
                },
            )

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
                        "accepted_records": len(completed_ids),
                        "generated_this_run": generated_this_run,
                        "current_chunk_index": current_chunk_index,
                        "current_chunk_record_count": current_chunk_record_count,
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
            "accepted_records": len(completed_ids),
            "current_chunk_index": current_chunk_index,
            "current_chunk_record_count": current_chunk_record_count,
        },
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the LiveCodeBench dataset loading script to run.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--backend",
        choices=("auto", "mlx", "vllm"),
        default="auto",
        help="Generation backend. auto uses MLX on macOS and vLLM elsewhere.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model path or Hugging Face model id. Defaults to the local MLX "
            "Qwen3-4B path when using the MLX backend."
        ),
    )
    parser.add_argument(
        "--model-path",
        dest="model",
        default=None,
        help="Deprecated alias for --model.",
    )
    parser.add_argument(
        "--model-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow custom model code when loading the generation model.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass enable_thinking to chat templates that support it.",
    )
    parser.add_argument(
        "--no-think-suffix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append /no_think to the user message for Qwen3-style models.",
    )
    parser.add_argument(
        "--code-only-sft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip thinking blocks and markdown fences before writing SFT targets.",
    )
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Accepted records per records/sft_messages chunk file.",
    )
    parser.add_argument(
        "--vllm-dtype",
        default="auto",
        help="vLLM dtype passed to LLM(...). Ignored by the MLX backend.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size. Use 1 for one A100.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory vLLM may use. Ignored by the MLX backend.",
    )
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
