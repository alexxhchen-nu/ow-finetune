"""Rule-based baseline mirroring current OW-Text deterministic logic."""

ACTIONS = ["ask", "probe", "transition", "end"]


def predict_action(turn: dict) -> str:
    """Predict action using rules similar to OW-Text's state machine."""
    # TODO: implement end / transition / probe / ask logic
    raise NotImplementedError
