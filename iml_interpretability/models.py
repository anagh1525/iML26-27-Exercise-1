"""Random Forest configurations for interpretability experiments."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return Random Forest models with different complexities."""

    configurations = [
        (1, 2),
        (5, 2),
        (20, 2),
        (5, 4),
        (20, 4),
        (50, 4),
        (20, 8),
        (50, 8),
    ]

    return {
        f"rf_{n_trees}_trees_depth_{depth}": RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=depth,
            random_state=seed,
            n_jobs=-1,
        )
        for n_trees, depth in configurations
    }