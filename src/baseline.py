"""Rule-based baseline mirroring OW-Text deterministic logic.

Usage:
    uv run python src/baseline.py data/dataset/val.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_ACTIONS = {"ask", "probe", "transition", "end"}

VAGUE_MARKERS = [
    "maybe", "perhaps", "i guess", "i think", "not sure", "kind of", "sort of",
    "a bit", "a little", "sometimes", "usually", "generally", "probably",
    "like", "you know", "i mean", "hmm", "嗯", "可能", "大概", "也许", "差不多",
    "有时候", "一般", "应该", "好像", "不太确定", "说不清", "不知道",
]

EMOTION_MARKERS = [
    "happy", "sad", "angry", "love", "hate", "excited", "worried", "afraid",
    "feel", "feeling", "emotion", "stress", "anxious", "depressed", "joy",
    "开心", "难过", "生气", "喜欢", "讨厌", "兴奋", "担心", "害怕", "压力", "焦虑",
    "高兴", "失望", "紧张", "感动", "痛苦",
]

ENDING_MARKERS = [
    "anything else", "last question", "finally", "wrap up", "conclude",
    "thank you for", "that's all", "end of", "还有什么", "最后一个问题", "总结",
    "感谢", "就到这里", "结束", "谢谢",
]


def probe_reasons(answer: str, question: str) -> list[str]:
    reasons = []
    lowered_answer = answer.lower()
    lowered_question = question.lower()

    if len(answer.strip()) < 30:
        reasons.append("short_answer")

    if any(marker in lowered_answer for marker in VAGUE_MARKERS):
        reasons.append("vague_answer")

    if any(marker in lowered_answer for marker in EMOTION_MARKERS):
        reasons.append("emotional_content")

    if "?" in answer or "？" in answer:
        reasons.append("respondent_asks_question")

    return reasons


def predict_action(record: dict) -> str:
    question = (record.get("question") or "").strip()
    answer = (record.get("answer") or "").strip()
    context = record.get("context", [])
    actions_so_far = [c.get("action") for c in context]

    lowered_question = question.lower()
    lowered_answer = answer.lower()

    if any(marker in lowered_question for marker in ENDING_MARKERS):
        return "end"

    reasons = probe_reasons(answer, question)
    if reasons:
        probe_count = sum(1 for a in actions_so_far if a == "probe")
        if probe_count < 2:
            return "probe"

    last_action = actions_so_far[-1] if actions_so_far else None
    if last_action == "probe":
        recent_answers = [c.get("answer", "").lower() for c in context[-2:]]
        current_lower = lowered_answer
        if any(
            current_lower[:50] == ra[:50]
            for ra in recent_answers
            if ra
        ):
            return "transition"

    return "ask"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based baseline classifier.")
    parser.add_argument("path", help="Path to dataset JSONL (train.jsonl or val.jsonl)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

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

    if not records:
        print("No valid records found", file=sys.stderr)
        sys.exit(1)

    predictions = [predict_action(r) for r in records]
    labels = [r.get("action", "") for r in records]

    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(records)
    accuracy = correct / total if total else 0

    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.3f}")
    print()

    action_counts = {}
    for l in labels:
        action_counts[l] = action_counts.get(l, 0) + 1
    print("Label distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")

    pred_counts = {}
    for p in predictions:
        pred_counts[p] = pred_counts.get(p, 0) + 1
    print("\nPrediction distribution:")
    for action, count in sorted(pred_counts.items()):
        print(f"  {action}: {count}")

    print("\nPer-action breakdown:")
    for action in sorted(VALID_ACTIONS):
        tp = sum(1 for p, l in zip(predictions, labels) if p == action and l == action)
        fp = sum(1 for p, l in zip(predictions, labels) if p == action and l != action)
        fn = sum(1 for p, l in zip(predictions, labels) if p != action and l == action)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print(f"  {action}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
