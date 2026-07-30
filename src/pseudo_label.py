"""Pseudo-label interview transcripts or session logs using an OpenAI-compatible LLM API.

Config priority:
    1. CLI: --provider, --base-url, --api-key, --model
    2. Env: LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    3. Prompt fallback if missing

Batch example:
    LLM_PROVIDER=custom \
    LLM_BASE_URL=http://localhost:20128/v1 \
    LLM_API_KEY=... \
    LLM_MODEL=claude-kimi \
    uv run python src/pseudo_label.py -o data/raw_uploads
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUPPORTED_EXTS = {".md", ".txt", ".json"}
BAD_MODEL_HINTS = ("tts", "image", "embed", "rerank", "whisper", "audio")

PROVIDERS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": "http://localhost:8000/v1",
}

PROMPT_TEMPLATE = """You are labeling a qualitative interview transcript chunk for an AI interviewer.

Chunk: {chunk_index}/{chunk_total}

Identify each interviewer turn in this chunk. For each interviewer turn, decide the action from one of:
- ask: ask a new main question
- probe: dig deeper into the current answer, clarify vague points, or follow up on emotional/new detail
- transition: move from the current topic to the next topic
- end: conclude the interview

Return a JSON array only. Each item must contain:
- question: interviewer exact question
- answer: respondent answer that follows
- action: one of [ask, probe, transition, end]
- reason: one short sentence

If no interviewer/respondent turn exists, return [].
No markdown fences. No extra text.

Transcript chunk:
---
{transcript}
---
"""


def normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url.rstrip("/")


def _int(value: str | None, default: int) -> int:
    return int(value) if value else default


def _temperature(value: str | None) -> int | float:
    if not value:
        return 1
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def select_provider() -> str:
    print("\nAvailable providers:")
    for i, name in enumerate(PROVIDERS, 1):
        print(f"  {i}. {name}")
    print("  0. custom")
    choice = input("\nSelect provider [1]: ").strip() or "1"
    try:
        idx = int(choice)
    except ValueError:
        idx = 1
    return "custom" if idx == 0 else list(PROVIDERS.keys())[idx - 1]


def fetch_models(base_url: str, api_key: str) -> list[str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{base_url}/models", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    models: list[str] = []
    for item in data.get("data", []):
        model_id = item.get("id")
        if not model_id:
            continue
        lowered = model_id.lower()
        if any(hint in lowered for hint in BAD_MODEL_HINTS):
            continue
        capabilities = item.get("capabilities") or {}
        if capabilities.get("imageOutput") or capabilities.get("audioOutput"):
            continue
        models.append(model_id)
    return models


def select_model(models: list[str]) -> str:
    if not models:
        return input("Model name: ").strip()
    print(f"\nAvailable models from API ({len(models)}):")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    choice = input("Select model [1]: ").strip() or "1"
    try:
        return models[int(choice) - 1]
    except (ValueError, IndexError):
        return models[0]


def model_order(preferred: str, available: list[str]) -> list[str]:
    ordered = [preferred] if preferred else []
    ordered.extend(model for model in available if model not in ordered)
    return ordered


def resolve_config(args: argparse.Namespace) -> tuple[str, str, str, list[str], int | float, int, int, int]:
    provider = args.provider or os.getenv("LLM_PROVIDER") or select_provider()

    base_url = args.base_url or os.getenv("LLM_BASE_URL")
    if not base_url:
        base_url = input(f"Base URL [{PROVIDERS[provider]}]: ").strip() or PROVIDERS[provider]
    base_url = normalize_base_url(base_url)

    api_key = args.api_key or os.getenv("LLM_API_KEY")
    if api_key is None:
        api_key = getpass.getpass("API key (leave empty if none): ").strip()

    available = fetch_models(base_url, api_key)
    model = args.model or os.getenv("LLM_MODEL")
    if not model:
        model = select_model(available) if sys.stdin.isatty() else (available[0] if available else "")
    models = model_order(model, available)
    if not models:
        raise RuntimeError("No model provided and /models returned none")

    temperature = _temperature(args.temperature or os.getenv("LLM_TEMPERATURE"))
    chunk_chars = _int(args.max_chars or os.getenv("LLM_MAX_CHARS"), 6000)
    max_tokens = _int(args.max_tokens or os.getenv("LLM_MAX_TOKENS"), 2000)
    timeout = _int(args.timeout or os.getenv("LLM_TIMEOUT"), 60)
    return provider, base_url, api_key, models, temperature, chunk_chars, max_tokens, timeout


def _decode_error(e: urllib.error.HTTPError) -> str:
    try:
        body = e.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return f"HTTP {e.code}: {e.reason}\n{body}".strip()


def parse_json_array(content: str) -> list[dict[str, Any]]:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Fix common LLM output issues
    content = content.replace("\\'", "'")  # Python-style escaping
    content = content.replace('\\"', '"')  # double-escaped quotes

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as first_error:
        decoder = json.JSONDecoder()
        last_error = first_error
        parsed = None
        for start, char in enumerate(content):
            if char != "[":
                continue
            try:
                candidate, _ = decoder.raw_decode(content[start:])
            except json.JSONDecodeError as e:
                last_error = e
                continue
            if isinstance(candidate, list):
                parsed = candidate
                break
        if parsed is None:
            preview = content[:1000].replace("\n", "\\n")
            raise RuntimeError(f"Could not parse JSON array: {last_error}. Preview: {preview}") from first_error

    if not isinstance(parsed, list):
        raise RuntimeError("LLM did not return a JSON array")
    return parsed


def call_llm(
    chunk: str,
    chunk_index: int,
    chunk_total: int,
    model: str,
    api_key: str,
    base_url: str,
    temperature: int | float,
    max_tokens: int,
    timeout: int,
) -> list[dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    transcript=chunk,
                    chunk_index=chunk_index,
                    chunk_total=chunk_total,
                ),
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace").lstrip()
            body, _ = json.JSONDecoder().raw_decode(raw_body)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM request failed: {_decode_error(e)}") from e

    try:
        content = body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected LLM response shape: {body}") from e
    return parse_json_array(content)


def load_transcript(path: Path) -> str:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "messages" in data:
            return "\n\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in data["messages"]
            )
        return json.dumps(data, ensure_ascii=False, indent=2)
    return path.read_text(encoding="utf-8")


def focus_transcript(text: str) -> str:
    for marker in ("**Detailed Transcript**", "Detailed Transcript", "详细访谈", "访谈原文"):
        idx = text.lower().find(marker.lower())
        if idx != -1:
            return text[idx:]
    return text


def split_chunks(text: str, chunk_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for start in range(0, len(paragraph), chunk_chars):
                chunks.append(paragraph[start : start + chunk_chars])
            continue

        next_len = current_len + len(paragraph) + 2
        if current and next_len > chunk_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = next_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


def find_files(path: Path, recursive: bool = False) -> list[Path]:
    files = path.rglob("*") if recursive else path.iterdir()
    return sorted(
        p
        for p in files
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and not p.name.lower().startswith("manifest.")
    )


def label_chunk(
    chunk: str,
    chunk_index: int,
    chunk_total: int,
    models: list[str],
    api_key: str,
    base_url: str,
    temperature: int | float,
    max_tokens: int,
    timeout: int,
) -> tuple[list[dict[str, Any]], str]:
    last_error: Exception | None = None
    for model in models:
        try:
            return call_llm(chunk, chunk_index, chunk_total, model, api_key, base_url, temperature, max_tokens, timeout), model
        except Exception as e:
            last_error = e
            print(f"    model failed {model}: {e}")
    raise RuntimeError(last_error or "all models failed")


def output_path_for(path: Path) -> Path:
    return LABELED_DIR / f"{path.stem}.pseudo_labeled.jsonl"


def process_file(
    path: Path,
    provider: str,
    models: list[str],
    api_key: str,
    base_url: str,
    temperature: int | float,
    chunk_chars: int,
    max_tokens: int,
    timeout: int,
    overwrite: bool,
) -> tuple[int, str, Path]:
    output = output_path_for(path)
    if output.exists() and not overwrite:
        print(f"  skip {path.name}: {output.name} already exists")
        return 0, "", output

    transcript = focus_transcript(load_transcript(path))
    chunks = split_chunks(transcript, chunk_chars)
    print(f"  file {path.name}: {len(chunks)} chunk(s)")

    all_labels: list[dict[str, Any]] = []
    preferred = ""
    for idx, chunk in enumerate(chunks, 1):
        print(f"    chunk {idx}/{len(chunks)}")
        try:
            labels, used_model = label_chunk(chunk, idx, len(chunks), models, api_key, base_url, temperature, max_tokens, timeout)
        except Exception as e:
            print(f"    chunk failed {idx}/{len(chunks)}: {e}")
            continue
        preferred = used_model
        models = [used_model] + [candidate for candidate in models if candidate != used_model]
        for record in labels:
            record["turn_index"] = len(all_labels) + 1
            record["chunk_index"] = idx
            record["source"] = str(path)
            record["provider"] = provider
            record["model"] = used_model
            all_labels.append(record)

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in all_labels:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  done {path.name}: {len(all_labels)} turns -> {output.name}")
    return len(all_labels), preferred, output


def write_summary(outputs: list[Path]) -> Path:
    action_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    file_counts: dict[str, int] = {}
    total = 0

    for output in outputs:
        count = 0
        if not output.exists():
            continue
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            count += 1
            total += 1
            action_counts[record.get("action", "unknown")] += 1
            model_counts[record.get("model", "unknown")] += 1
        file_counts[output.name] = count

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / "pseudo_label_summary.md"
    lines = [
        "# Pseudo Label Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Total turns: {total}",
        "",
        "## Files",
    ]
    lines.extend(f"- `{name}`: {count} turns" for name, count in sorted(file_counts.items()))
    lines.extend(["", "## Actions"])
    lines.extend(f"- `{action}`: {count}" for action, count in sorted(action_counts.items()))
    lines.extend(["", "## Models"])
    lines.extend(f"- `{model}`: {count}" for model, count in sorted(model_counts.items()))
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label transcripts with an LLM.")
    parser.add_argument("path", help="Path to a transcript file or directory")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing labeled files")
    parser.add_argument("--provider", help="LLM provider (or env LLM_PROVIDER)")
    parser.add_argument("--base-url", help="API base URL (or env LLM_BASE_URL)")
    parser.add_argument("--api-key", help="API key (or env LLM_API_KEY)")
    parser.add_argument("--model", help="Preferred model (or env LLM_MODEL)")
    parser.add_argument("--temperature", help="Sampling temperature (or env LLM_TEMPERATURE). Default: 1")
    parser.add_argument("--max-chars", help="Chunk character cap (or env LLM_MAX_CHARS). Default: 6000")
    parser.add_argument("--max-tokens", help="Output token cap (or env LLM_MAX_TOKENS). Default: 2000")
    parser.add_argument("--timeout", help="HTTP timeout seconds (or env LLM_TIMEOUT). Default: 60")
    parser.add_argument("--limit", type=int, help="Only process first N files in a directory")
    args = parser.parse_args()

    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    if target.is_dir():
        files = find_files(target, recursive=args.recursive)
        if args.limit:
            files = files[: args.limit]
    else:
        if target.suffix.lower() not in SUPPORTED_EXTS:
            print(f"Unsupported file type: {target.suffix}. Supported: {SUPPORTED_EXTS}", file=sys.stderr)
            sys.exit(1)
        files = [target]

    if not files:
        print(f"No supported files found. Supported: {', '.join(sorted(SUPPORTED_EXTS))}")
        sys.exit(0)

    provider, base_url, api_key, models, temperature, chunk_chars, max_tokens, timeout = resolve_config(args)
    print(f"\nFound {len(files)} file(s)")
    print(f"Model fallback count: {len(models)}; first: {models[0]}")
    print(f"Config: temperature={temperature}, chunk_chars={chunk_chars}, max_tokens={max_tokens}, timeout={timeout}")

    total = 0
    outputs: list[Path] = []
    for path in files:
        try:
            turns, preferred, output = process_file(path, provider, models, api_key, base_url, temperature, chunk_chars, max_tokens, timeout, args.overwrite)
            total += turns
            outputs.append(output)
            if preferred:
                models = [preferred] + [candidate for candidate in models if candidate != preferred]
        except Exception as e:
            print(f"  failed {path.name}: {e}", file=sys.stderr)

    report = write_summary(outputs)
    print(f"\nTotal new labeled turns: {total}")
    print(f"Summary: {report}")


if __name__ == "__main__":
    main()
