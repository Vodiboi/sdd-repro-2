# Qwen3-4B SSD Dataset

This folder is the handoff location for the generated simple self-distillation
dataset from `../data 2.csv`.

Run the full generator from the project root with Python 3.10:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py
```

For a small smoke run:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 generate_ssd_dataset.py --stop-after 3
```

Rerun the same command to resume. Existing accepted examples are skipped by
their deterministic IDs.

Expected files:

- `records.jsonl`: full records with metadata, token counts, generation settings,
  and system/user/assistant messages.
- `sft_messages.jsonl`: minimal chat-format records containing only `messages`.
- `skipped_rows.jsonl`: empty prompts, duplicate prompt mappings, filtered
  generations, and generation failures.
- `manifest.json`: current counts and generation configuration.
- `progress_state.json`: last checkpoint status.

Generation uses `/Users/aayanarish/models/Qwen3-4B-4bit` with MLX, one sample
per unique non-empty problem statement, `temperature=1.6`, `top_k=20`,
`top_p=0.8`, and a 32,768-token prompt+output cap. No code is executed and no
correctness filtering is applied.
