"""Evaluation metrics for the interaction-logic classifier."""


def accuracy(predictions: list[str], labels: list[str]) -> float:
    """Fraction of correctly predicted actions."""
    raise NotImplementedError


def per_action_f1(predictions: list[str], labels: list[str]) -> dict[str, float]:
    """F1 score per action."""
    raise NotImplementedError
