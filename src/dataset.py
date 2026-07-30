"""Merge labeled JSONL files into a training dataset.

Usage:
    uv run python src/dataset.py                  # default: merge data/labeled/*.jsonl
    uv run python src/dataset.py --input data/labeled --output data/dataset
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
VALID_ACTIONS = {"ask", "probe", "transition", "end"}


def load_labeled(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        action = record.get("action", "").strip().lower()
        if action not in VALID_ACTIONS:
            continue
        question = (record.get("question") or "").strip()
        answer = (record.get("answer") or "").strip()
        if not question:
            continue
        records.append({
            "question": question,
            "answer": answer,
            "action": action,
            "reason": (record.get("reason") or "").strip(),
            "source": record.get("source", ""),
            "model": record.get("model", ""),
            "turn_index": record.get("turn_index", 0),
            "chunk_index": record.get("chunk_index", 0),
        })
    return records


def build_context(records: list[dict], context_size: int = 3) -> list[dict]:
    """Build training examples with preceding turns as context."""
    examples = []
    for i, record in enumerate(records):
        context = []
        for j in range(max(0, i - context_size), i):
            context.append({
                "question": records[j]["question"],
                "answer": records[j]["answer"],
                "action": records[j]["action"],
            })
        examples.append({
            "context": context,
            "question": record["question"],
            "answer": record["answer"],
            "action": record["action"],
            "reason": record["reason"],
            "source": record["source"],
            "model": record["model"],
        })
    return examples


def split_dataset(examples: list[dict], val_ratio: float = 0.15, seed: int = 42) -> tuple[list[dict], list[dict]]:
    random.seed(seed)
    shuffled = list(examples)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge labeled JSONL into training dataset.")
    parser.add_argument("--input", type=Path, default=LABELED_DIR, help="Directory with .jsonl files")
    parser.add_argument("--output", type=Path, default=DATASET_DIR, help="Output directory")
    parser.add_argument("--context-size", type=int, default=3, help="Number of preceding turns as context")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    args = parser.parse_args()

    jsonl_files = sorted(args.input.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    all_records: list[dict] = []
    for path in jsonl_files:
        records = load_labeled(path)
        print(f"  {path.name}: {len(records)} valid records")
        all_records.extend(records)

    if not all_records:
        print("No valid records found", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal raw records: {len(all_records)}")

    examples = build_context(all_records, context_size=args.context_size)
    train, val = split_dataset(examples, val_ratio=args.val_ratio)

    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    val_path = args.output / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    action_counts = {}
    for ex in examples:
        action_counts[ex["action"]] = action_counts.get(ex["action"], 0) + 1

    print(f"Train: {len(train)} | Val: {len(val)}")
    print(f"Actions: {action_counts}")
    print(f"Output: {train_path} / {val_path}")


if __name__ == "__main__":
    main()
