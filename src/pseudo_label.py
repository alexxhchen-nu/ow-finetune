"""Pseudo-label interview transcripts or session logs using an LLM.

Usage:
    uv run python src/pseudo_label.py <path-to-transcript.md>
    uv run python src/pseudo_label.py <path-to-session.json>

The script will interactively ask for provider, API key, base URL, then query the
API's /models endpoint so you can pick a model by number.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"

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


def select_provider() -> tuple[str, str]:
    """Interactively choose provider and base URL."""
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
        provider = "custom"
    else:
        provider = list(PROVIDERS.keys())[idx - 1]

    default_base = PROVIDERS[provider]["default_base"]
    base_url = input(f"Base URL [{default_base}]: ").strip() or default_base
    return provider, normalize_base_url(base_url)


def get_api_key() -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label a transcript with LLM.")
    parser.add_argument("path", help="Path to transcript .md/.txt/.json")
    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    provider, base_url = select_provider()
    api_key = get_api_key()

    models = fetch_models(base_url, api_key)
    if models:
        print(f"\nFetched {len(models)} models from {base_url}/models")
    else:
        print(f"\nCould not fetch models from {base_url}/models; you can enter one manually.")
    model = select_model(models)

    transcript = load_transcript(path)
    if len(transcript) > 12000:
        print("Transcript is long; sending first ~12000 characters to the LLM.")
        transcript = transcript[:12000]

    print(f"\nLabeling with {provider}/{model} ...")
    labels = call_llm(transcript, model, api_key, base_url)

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    output = LABELED_DIR / f"{path.stem}.pseudo_labeled.jsonl"
    with output.open("w", encoding="utf-8") as f:
        for record in labels:
            record["source"] = str(path)
            record["provider"] = provider
            record["model"] = model
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(labels)} labeled turns to {output}")


if __name__ == "__main__":
    main()
