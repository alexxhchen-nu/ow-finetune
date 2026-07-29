"""Ingest arbitrary files into the project as raw resources.

Usage:
    uv run python src/ingest.py /path/to/file.docx /path/to/session.json
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_uploads"
MANIFEST_CSV = RAW_DIR / "manifest.csv"
EXTRACTABLE = {".docx", ".pdf", ".txt", ".md", ".markdown", ".json", ".csv"}
ResourceType = Literal["session_json", "interview_transcript", "design_doc", "report", "csv_data", "unknown"]


def classify(path: Path, content: str) -> tuple[ResourceType, str]:
    """Guess resource type from file content and name."""
    suffix = path.suffix.lower()
    name_lower = path.stem.lower()

    if suffix == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return "unknown", "invalid JSON"
        if isinstance(data, dict) and "messages" in data and "state" in data:
            return "session_json", "OW-Text session export"
        return "unknown", "JSON without OW-Text session shape"

    if suffix == ".csv":
        return "csv_data", "tabular data"

    if "interview" in name_lower or "访谈" in name_lower:
        return "interview_transcript", "interview-like filename"

    if "framework" in name_lower or "logic" in name_lower or "design" in name_lower:
        return "design_doc", "design-like filename"

    if "report" in name_lower or "报告" in name_lower:
        return "report", "report-like filename"

    if "transcript" in name_lower or "q&a" in name_lower or "对话" in name_lower:
        return "interview_transcript", "transcript-like filename"

    # Heuristic: transcripts usually have speaker labels or question marks early.
    lines = content.splitlines()[:50]
    joined = "\n".join(lines).lower()
    if any(label in joined for label in ("speaker 1", "speaker 2", "interviewer", "interviewee", "q:", "a:")):
        return "interview_transcript", "speaker labels detected"

    return "unknown", "fallback"


def extract_content(path: Path) -> str:
    """Return a readable text/markdown representation of the file."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False, indent=2)

    if suffix == ".csv":
        return path.read_text(encoding="utf-8")

    if suffix == ".docx":
        result = subprocess.run(
            ["pandoc", "-f", "docx", "-t", "markdown", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pandoc failed for {path}: {result.stderr}")
        return result.stdout

    if suffix == ".pdf":
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftotext failed for {path}: {result.stderr}")
        return result.stdout

    raise ValueError(f"Unsupported file type: {suffix}")


def ingest_file(path: Path) -> dict:
    """Ingest a single file into data/raw_uploads/ and update the manifest."""
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix not in EXTRACTABLE:
        raise ValueError(f"Unsupported extension: {suffix}. Supported: {EXTRACTABLE}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Copy original.
    dest = RAW_DIR / path.name
    shutil.copy2(path, dest)

    # Extract text.
    extracted = extract_content(dest)
    extracted_path = RAW_DIR / f"{dest.stem}.extracted.md"
    extracted_path.write_text(extracted, encoding="utf-8")

    # Classify.
    resource_type, note = classify(dest, extracted)

    record = {
        "filename": dest.name,
        "source": str(path),
        "type": resource_type,
        "format": suffix.lstrip("."),
        "extracted": extracted_path.name,
        "notes": note,
    }

    _append_manifest(record)
    return record


def _append_manifest(record: dict) -> None:
    """Append one row to manifest.csv, creating it with headers if needed."""
    fieldnames = ["filename", "source", "type", "format", "extracted", "notes"]
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST_CSV.exists()

    with MANIFEST_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw files into the project.")
    parser.add_argument("paths", nargs="+", help="Files to ingest")
    args = parser.parse_args()

    for path_str in args.paths:
        path = Path(path_str).expanduser().resolve()
        try:
            record = ingest_file(path)
            print(f"✓ {record['type']}: {record['filename']} -> {record['extracted']}")
        except Exception as e:
            print(f"✗ {path}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
