# Qwen3-4B SSD Dataset Generation

This repository prepares a simple self-distillation (SSD) dataset for Qwen3-4B.
It starts from the Hugging Face `livecodebench/code_generation_lite` dataset,
asks Qwen3-4B to generate one Python solution per unique prompt, and writes
chat-format JSONL that can be used for supervised fine-tuning.

The implementation follows the dataset-generation portion of the SSD paper's
recipe in spirit: sample model outputs, avoid correctness verification, and
avoid code execution. For this reproduction, the SFT handoff is intentionally
code-only so Qwen thinking traces and explanations do not become training
targets.

## Repository Contents

- `generate_ssd_dataset.py` - resumable MLX/vLLM generation script.
- `ssd_qwen3_4b_lcb_dataset/` - output handoff folder for generated records,
  manifests, and skipped-row logs.
- `old_data_gen/` - previous CSV-based source and generated-data artifacts,
  retained separately from the active LiveCodeBench workflow.
- `a.ipynb` - small local Qwen/MLX notebook used during setup.

Local smoke-test outputs and Python caches are intentionally ignored by Git.

## Requirements

The script supports two generation backends:

- `mlx` for macOS Apple Silicon with MLX.
- `vllm` for Linux CUDA machines such as an NVIDIA A100 or RTX A5000.

On this Mac, the default MLX-format Qwen3-4B model is:

```text
/Users/aayanarish/models/Qwen3-4B-4bit
```

The tested Python interpreter is:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10
```

Required Mac packages include `mlx`, `mlx-lm`, `transformers`, `datasets`, and
their dependencies. MLX needs access to Metal, so generation may fail in
headless or sandboxed shells that cannot see the GPU.

For CUDA generation, use Linux with Python 3.10 and install CUDA-compatible
`vllm`, `datasets`, and `transformers`. Use a Hugging Face-format Qwen3-4B model
folder or model id. The local MLX 4-bit folder above will not run on CUDA GPUs.
Use the official vLLM GPU installation notes for the current CUDA wheel command:
<https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>.

## Prompt Source

The active prompt source is:

```text
livecodebench/code_generation_lite
```

By default the script loads the `test` split with `trust_remote_code=True`,
which LiveCodeBench requires for its dataset loader. Each prompt uses
`question_content`; if `starter_code` is present, it is appended in a fenced
Python block. Public and private test cases are kept out of the prompt and are
not used for filtering.

LiveCodeBench shards such as `test.jsonl`, `test2.jsonl`, and `test3.jsonl` are
downloaded by Hugging Face `datasets` into the machine's HF cache, not into this
repository. Set `HF_HOME` before running if you want that cache on a specific
disk.

## Generation Recipe

The default script settings are:

- backend: `auto` (`mlx` on macOS, `vllm` elsewhere)
- Mac model: `/Users/aayanarish/models/Qwen3-4B-4bit`
- CUDA model: explicit `--model /path/to/HF-format-Qwen3-4B`
- samples per prompt: `1`
- accepted-record chunk size: `100`
- context cap: `32768` prompt + output tokens
- temperature: `1.6`
- top-k: `20`
- top-p: `0.8`
- Qwen thinking: disabled by default (`enable_thinking=False` when supported,
  plus `/no_think` suffix)
- SFT target cleanup: strip thinking blocks and markdown fences by default
- filtering: skip empty outputs and single-line stubs only
- verification: no test execution and no correctness filtering

The dataset is audited before generation. Empty prompts are skipped, and
duplicate prompts are generated only once using whitespace-normalized
deduplication. Duplicate mappings are written to `skipped_rows.jsonl`.

## Usage

From the repository root, prepare the output folder and manifests without loading
the model:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py --backend mlx --prepare-only
```

Run a small Mac/MLX smoke test:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py --backend mlx --output-dir ssd_qwen3_4b_dataset_dryrun --stop-after 3 --max-context-tokens 2048 --seed 1
```

Run the full Mac/MLX generation job:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py --backend mlx
```

Run a small CUDA/vLLM smoke test:

```bash
python3.10 generate_ssd_dataset.py \
  --backend vllm \
  --model /path/to/HF-format-Qwen3-4B \
  --output-dir ssd_qwen3_4b_lcb_dataset_cuda_dryrun \
  --stop-after 3 \
  --max-context-tokens 2048
```

Run the full CUDA/vLLM generation job:

```bash
python3.10 generate_ssd_dataset.py \
  --backend vllm \
  --model /path/to/HF-format-Qwen3-4B \
  --output-dir ssd_qwen3_4b_lcb_dataset_cuda
```

The job is append-only and resumable. Accepted records are fsynced one at a time
and grouped into chunk files containing 100 accepted records by default. If the
job is interrupted, rerun the same command; completed record IDs are discovered
from existing chunks and skipped.

Validate an output folder:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py --backend mlx --validate-only
```

## Output Files

The main output folder is `ssd_qwen3_4b_lcb_dataset/`.

- `records/records_*.jsonl` - chunked full records with source metadata, chat
  messages, token counts, finish reason, elapsed time, and generation settings.
- `sft_messages/sft_messages_*.jsonl` - chunked minimal SFT records containing
  only `messages`.
- `skipped_rows.jsonl` - empty rows, duplicate mappings, filtered generations,
  context-budget skips, and generation failures.
- `manifest.json` - current counts, paths, and generation settings.
- `progress_state.json` - last checkpoint status for long runs.
- `README.md` - output-folder-specific handoff notes.

Older output folders with root-level `records.jsonl` and `sft_messages.jsonl`
are still supported for validation and resume.

Each accepted SFT record has three chat messages:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "...problem statement..."},
    {"role": "assistant", "content": "...generated solution..."}
  ]
}
```

## Notes

The full job can take a long time. A dry run with a 2,048-token context cap
showed that examples may generate until the length limit, so the default 32,768
token cap should be treated as a long batch workload rather than an interactive
notebook cell.

The generated outputs are intentionally unverified. This is by design for SSD
data synthesis and should not be interpreted as a collection of correct
solutions.
