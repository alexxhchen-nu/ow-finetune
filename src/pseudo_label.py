"""Pseudo-label interview transcripts or session logs using an LLM.

Configuration priority (highest first):
    1. CLI arguments: --provider, --base-url, --api-key, --model
    2. Environment variables: LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    3. Interactive prompts

Usage:
    # single file (env-based, no prompts)
    LLM_PROVIDER=custom LLM_BASE_URL=http://localhost:20128/v1 \
    LLM_MODEL=claude-kimi uv run python src/pseudo_label.py transcript.md

    # batch directory
    uv run python src/pseudo_label.py data/raw_uploads

    # batch directory recursively + overwrite
    uv run python src/pseudo_label.py -r -o data/raw_uploads
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
SUPPORTED_EXTS = {".md", ".txt", ".json"}

PROVIDERS = {
    "openai": {
        "default_base": "https://api.openai.com/v1",
    },
    "deepseek": {
        "default_base": "https://api.deepseek.com/v1",
    },
    "openrouter": {
        "default_base": "https://openrouter.ai/api/v1",
    },
    "custom": {
        "default_base": "http://localhost:8000/v1",
    },
}

PROMPT_TEMPLATE = """You are labeling a qualitative interview transcript for an AI interviewer.

The transcript may contain interviewer questions and respondent answers. Identify each interviewer turn and decide the interviewer's next action from one of:
- ask: ask a new main question
- probe: dig deeper into the current answer, clarify vague points, or follow up on emotional/new detail
- transition: move from the current topic to the next topic
- end: conclude the interview

For each interviewer turn, return:
- turn_index: 1-based index
- question: the interviewer's exact question
- answer: the respondent's answer that follows
- action: one of [ask, probe, transition, end]
- reason: one short sentence explaining why

Return ONLY a valid JSON array. Do not include markdown code fences or extra text.

Transcript:
---
{transcript}
---
"""


def normalize_base_url(url: str) -> str:
    """Treat the URL as the API base, stripping a trailing /chat/completions if present."""
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url.rstrip("/")


def select_provider() -> str:
    print("\nAvailable providers:")
    for i, name in enumerate(PROVIDERS, 1):
        print(f"  {i}. {name}")
    print("  0. custom (enter base URL manually)")

    choice = input("\nSelect provider [1]: ").strip() or "1"
    try:
        idx = int(choice)
    except ValueError:
        idx = 1

    if idx == 0:
        return "custom"
    return list(PROVIDERS.keys())[idx - 1]


def ask_base_url(provider: str) -> str:
    default = PROVIDERS[provider]["default_base"]
    return input(f"Base URL [{default}]: ").strip() or default


def ask_api_key() -> str:
    return getpass.getpass("API key (leave empty if none): ").strip()


def fetch_models(base_url: str, api_key: str) -> list[str]:
    """Try to GET /models from the API. Return empty list on failure."""
    models_url = f"{base_url}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["id"] for m in data.get("data", []) if "id" in m]
    except Exception:
        return []


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


def resolve_config(args: argparse.Namespace) -> tuple[str, str, str, str]:
    """Return (provider, base_url, api_key, model) using args -> env -> prompt."""
    # provider
    provider = args.provider or os.getenv("LLM_PROVIDER")
    if not provider:
        provider = select_provider()

    # base_url
    base_url = args.base_url or os.getenv("LLM_BASE_URL")
    if not base_url:
        base_url = ask_base_url(provider)
    base_url = normalize_base_url(base_url)

    # api_key
    api_key = args.api_key or os.getenv("LLM_API_KEY")
    if api_key is None:
        api_key = ask_api_key()

    # model
    model = args.model or os.getenv("LLM_MODEL")
    if not model:
        models = fetch_models(base_url, api_key)
        if models:
            print(f"\nFetched {len(models)} models from {base_url}/models")
        else:
            print(f"\nCould not fetch models from {base_url}/models; enter one manually.")
        model = select_model(models)

    return provider, base_url, api_key, model


def _decode_error(e: urllib.error.HTTPError) -> str:
    try:
        body = e.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return f"HTTP {e.code}: {e.reason}\n{body}".strip()


def call_llm(transcript: str, model: str, api_key: str, base_url: str) -> list[dict[str, Any]]:
    chat_url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(transcript=transcript)}],
        "temperature": 0.3,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        chat_url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM request failed: {_decode_error(e)}") from e

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected LLM response shape: {body}") from e

    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    labels = json.loads(content)
    if not isinstance(labels, list):
        raise RuntimeError("LLM did not return a JSON array")
    return labels


def load_transcript(path: Path) -> str:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "messages" in data:
            # OW-Text session JSON
            return "\n\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in data["messages"]
            )
        return json.dumps(data, ensure_ascii=False, indent=2)
    return path.read_text(encoding="utf-8")


def find_files(path: Path, recursive: bool = False) -> list[Path]:
    if recursive:
        files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    else:
        files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def process_file(path: Path, provider: str, model: str, api_key: str, base_url: str, overwrite: bool) -> int:
    output = LABELED_DIR / f"{path.stem}.pseudo_labeled.jsonl"
    if output.exists() and not overwrite:
        print(f"  skip {path.name}: {output.name} already exists")
        return 0

    transcript = load_transcript(path)
    if len(transcript) > 12000:
        print(f"  note {path.name}: long, sending first ~12000 characters")
        transcript = transcript[:12000]

    labels = call_llm(transcript, model, api_key, base_url)

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in labels:
            record["source"] = str(path)
            record["provider"] = provider
            record["model"] = model
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  done {path.name}: {len(labels)} turns -> {output.name}")
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label transcripts with an LLM.")
    parser.add_argument("path", help="Path to a transcript file or a directory of files")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing labeled files")
    parser.add_argument("--provider", help="LLM provider (or env LLM_PROVIDER)")
    parser.add_argument("--base-url", help="API base URL (or env LLM_BASE_URL)")
    parser.add_argument("--api-key", help="API key (or env LLM_API_KEY)")
    parser.add_argument("--model", help="Model name (or env LLM_MODEL)")
    args = parser.parse_args()

    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    if target.is_dir():
        files = find_files(target, recursive=args.recursive)
        if not files:
            print(f"No supported files ({', '.join(sorted(SUPPORTED_EXTS))}) found in {target}")
            sys.exit(0)
        print(f"\nFound {len(files)} files to label:\n" + "\n".join(f"  - {f.name}" for f in files))
    else:
        if target.suffix.lower() not in SUPPORTED_EXTS:
            print(f"Unsupported file type: {target.suffix}. Supported: {SUPPORTED_EXTS}", file=sys.stderr)
            sys.exit(1)
        files = [target]

    provider, base_url, api_key, model = resolve_config(args)

    print(f"\nLabeling {len(files)} file(s) with {provider}/{model} ...")
    total = 0
    for path in files:
        try:
            total += process_file(path, provider, model, api_key, base_url, args.overwrite)
        except Exception as e:
            print(f"  failed {path.name}: {e}", file=sys.stderr)

    print(f"\nTotal labeled turns: {total}")


if __name__ == "__main__":
    main()
