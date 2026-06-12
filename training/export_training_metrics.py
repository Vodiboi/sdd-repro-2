#!/usr/bin/env python3
"""Export Hugging Face Trainer loss history from checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("training_outputs/qwen3_4b_lora")


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return -1


def trainer_state_paths(output_dir: Path) -> list[Path]:
    paths = []
    root_state = output_dir / "trainer_state.json"
    if root_state.exists():
        paths.append(root_state)
    for checkpoint in sorted(output_dir.glob("checkpoint-*"), key=checkpoint_step):
        state_path = checkpoint / "trainer_state.json"
        if state_path.exists():
            paths.append(state_path)
    return paths


def load_best_state(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    states = []
    for path in trainer_state_paths(output_dir):
        state = json.loads(path.read_text(encoding="utf-8"))
        log_history = state.get("log_history", [])
        states.append((len(log_history), checkpoint_step(path.parent), path, state))
    if not states:
        raise FileNotFoundError(f"No trainer_state.json files found under {output_dir}")
    _, _, path, state = max(states, key=lambda item: (item[0], item[1]))
    return path, state


def metric_rows(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in log_history:
        if "loss" not in entry and "eval_loss" not in entry:
            continue
        rows.append(
            {
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "loss": entry.get("loss"),
                "eval_loss": entry.get("eval_loss"),
                "learning_rate": entry.get("learning_rate"),
                "grad_norm": entry.get("grad_norm"),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, Any]], path: Path) -> bool:
    loss_points = [(row["step"], row["loss"]) for row in rows if row.get("step") is not None and row.get("loss") is not None]
    eval_points = [
        (row["step"], row["eval_loss"])
        for row in rows
        if row.get("step") is not None and row.get("eval_loss") is not None
    ]
    if not loss_points and not eval_points:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if loss_points:
        steps, losses = zip(*loss_points)
        ax.plot(steps, losses, label="train loss", linewidth=1.5)
    if eval_points:
        steps, losses = zip(*eval_points)
        ax.plot(steps, losses, label="eval loss", linewidth=1.5)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def export_metrics(output_dir: Path, metrics_dir: Path | None) -> dict[str, Any]:
    state_path, state = load_best_state(output_dir)
    rows = metric_rows(state.get("log_history", []))
    if not rows:
        raise ValueError(f"{state_path} does not contain loss entries")

    metrics_dir = metrics_dir or output_dir / "metrics"
    csv_path = metrics_dir / "training_metrics.csv"
    plot_path = metrics_dir / "loss.png"
    write_csv(rows, csv_path)

    plot_written = False
    try:
        plot_written = write_plot(rows, plot_path)
    except ImportError:
        plot_path = None

    return {
        "source": str(state_path),
        "rows": len(rows),
        "last_step": rows[-1].get("step"),
        "last_epoch": rows[-1].get("epoch"),
        "last_loss": rows[-1].get("loss"),
        "csv": str(csv_path),
        "plot": str(plot_path) if plot_written and plot_path is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        print(json.dumps(export_metrics(args.output_dir, args.metrics_dir), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
