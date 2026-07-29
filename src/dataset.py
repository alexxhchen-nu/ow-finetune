"""Parse OW-Text session JSON into turn-level training examples."""
from pathlib import Path
from typing import Iterator


def load_session(path: Path | str) -> dict:
    """Load a single OW-Text session export."""
    raise NotImplementedError


def sessions_to_turns(session: dict) -> Iterator[dict]:
    """Yield one record per assistant turn with ground-truth action."""
    raise NotImplementedError


def build_dataset(raw_dir: Path | str, output_path: Path | str) -> None:
    """Convert all raw sessions into a labeled dataset."""
    raise NotImplementedError
