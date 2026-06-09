# Qwen3-4B SSD Dataset Generation

This repository prepares a simple self-distillation (SSD) dataset for a local
Qwen3-4B model. It starts from the Hugging Face
`livecodebench/code_generation_lite` dataset, asks the same local Qwen3-4B
model to generate one Python solution per unique prompt, and writes chat-format
JSONL that can be used for supervised fine-tuning.

The implementation follows the dataset-generation portion of the SSD paper's
recipe: sample raw model outputs, avoid correctness verification, avoid code
execution, and apply only minimal degeneracy filtering.

## Repository Contents

- `generate_ssd_dataset.py` - resumable MLX generation script.
- `ssd_qwen3_4b_lcb_dataset/` - output handoff folder for generated records,
  manifests, and skipped-row logs.
- `old_data_gen/` - previous CSV-based source and generated-data artifacts,
  retained separately from the active LiveCodeBench workflow.
- `a.ipynb` - small local Qwen/MLX notebook used during setup.

Local smoke-test outputs and Python caches are intentionally ignored by Git.

## Requirements

This project expects a macOS Apple Silicon environment with MLX available and a
local MLX-format Qwen3-4B model at:

```text
/Users/aayanarish/models/Qwen3-4B-4bit
```

The tested Python interpreter is:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10
```

Required Python packages include `mlx`, `mlx-lm`, `transformers`, and their
dependencies. MLX needs access to Metal, so generation may fail in headless or
sandboxed shells that cannot see the GPU.

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

## Generation Recipe

The default script settings are:

- model: `/Users/aayanarish/models/Qwen3-4B-4bit`
- samples per prompt: `1`
- context cap: `32768` prompt + output tokens
- temperature: `1.6`
- top-k: `20`
- top-p: `0.8`
- filtering: skip empty outputs and single-line stubs only
- verification: no test execution and no correctness filtering

The dataset is audited before generation. Empty prompts are skipped, and
duplicate prompts are generated only once using whitespace-normalized
deduplication. Duplicate mappings are written to `skipped_rows.jsonl`.

## Usage

From the repository root, prepare the output folder and manifests without loading
the model:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py --prepare-only
```

Run a small smoke test:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py --output-dir ssd_qwen3_4b_dataset_dryrun --stop-after 3 --max-context-tokens 2048 --seed 1
```

Run the full generation job:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py
```

The job is append-only and resumable. If it is interrupted, rerun the same
command; completed record IDs are skipped.

Validate an output folder:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py --validate-only
```

## Output Files

The main output folder is `ssd_qwen3_4b_lcb_dataset/`.

- `records.jsonl` - full records with source metadata, chat messages, token
  counts, finish reason, elapsed time, and generation settings.
- `sft_messages.jsonl` - minimal SFT records containing only `messages`.
- `skipped_rows.jsonl` - empty rows, duplicate mappings, filtered generations,
  context-budget skips, and generation failures.
- `manifest.json` - current counts, paths, and generation settings.
- `progress_state.json` - last checkpoint status for long runs.
- `README.md` - output-folder-specific handoff notes.

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
