"""Evaluation metrics for interaction-logic classifiers.

Usage:
    uv run python src/evaluate.py predictions.jsonl labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_ACTIONS = {"ask", "probe", "transition", "end"}


def accuracy(predictions: list[str], labels: list[str]) -> float:
    if not labels:
        return 0.0
    return sum(1 for p, l in zip(predictions, labels) if p == l) / len(labels)


def per_action_metrics(predictions: list[str], labels: list[str]) -> dict[str, dict[str, float]]:
    results = {}
    for action in sorted(VALID_ACTIONS):
        tp = sum(1 for p, l in zip(predictions, labels) if p == action and l == action)
        fp = sum(1 for p, l in zip(predictions, labels) if p == action and l != action)
        fn = sum(1 for p, l in zip(predictions, labels) if p != action and l == action)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[action] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return results


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classifier predictions.")
    parser.add_argument("predictions", help="Path to predictions JSONL")
    parser.add_argument("labels", help="Path to labels JSONL (ground truth)")
    args = parser.parse_args()

    pred_records = load_jsonl(Path(args.predictions))
    label_records = load_jsonl(Path(args.labels))

    if len(pred_records) != len(label_records):
        print(f"Warning: predictions ({len(pred_records)}) != labels ({len(label_records)})", file=sys.stderr)

    predictions = [r.get("action", "") for r in pred_records]
    labels = [r.get("action", "") for r in label_records]

    print(f"Accuracy: {accuracy(predictions, labels):.3f}")
    print()
    metrics = per_action_metrics(predictions, labels)
    for action, m in metrics.items():
        print(f"  {action}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")


if __name__ == "__main__":
    main()
