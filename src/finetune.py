"""Train a simple classifier (TF-IDF + Logistic Regression) on pseudo-labeled data.

Usage:
    uv run python src/finetune.py
    uv run python src/finetune.py --input data/dataset --output models/tfidf_lr
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"

LABEL2ID = {"ask": 0, "probe": 1, "transition": 2, "end": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


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


def build_text(record: dict) -> str:
    context = record.get("context", [])
    parts = []
    for turn in context[-2:]:
        parts.append(f"Q: {turn.get('question', '')} A: {turn.get('answer', '')}")
    parts.append(f"Q: {record.get('question', '')} A: {record.get('answer', '')}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + LR classifier.")
    parser.add_argument("--input", type=Path, default=DATASET_DIR, help="Dataset directory")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "tfidf_lr", help="Output model directory")
    args = parser.parse_args()

    train_path = args.input / "train.jsonl"
    val_path = args.input / "val.jsonl"
    if not train_path.is_file() or not val_path.is_file():
        print(f"Missing {train_path} or {val_path}", file=sys.stderr)
        sys.exit(1)

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    print(f"Train: {len(train_records)} | Val: {len(val_records)}")

    X_train = [build_text(r) for r in train_records]
    y_train = [LABEL2ID[r["action"]] for r in train_records]
    X_val = [build_text(r) for r in val_records]
    y_val = [LABEL2ID[r["action"]] for r in val_records]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    print("\nClassification Report:")
    present_labels = sorted(set(y_val) | set(y_pred))
    target_names = [ID2LABEL[i] for i in present_labels]
    print(classification_report(y_val, y_pred, labels=present_labels, target_names=target_names, zero_division=0))

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to: {model_path}")

    label_dist = {}
    for r in train_records:
        label_dist[r["action"]] = label_dist.get(r["action"], 0) + 1
    print(f"Train label distribution: {label_dist}")


if __name__ == "__main__":
    main()
