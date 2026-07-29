"""Build prompt/state representation for a single turn."""
from typing import Any


def build_turn_features(turn: dict, session: dict) -> dict[str, Any]:
    """Return a structured feature dict from session state + transcript."""
    raise NotImplementedError
