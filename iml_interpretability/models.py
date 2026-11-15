
"""Inherently interpretable model candidates.

Defines Random Forest and Gradient Boosting classifiers with different
numbers of trees and maximum depths to investigate interpretability.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_random_forest_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return Random Forest classifiers with different complexities."""

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


def get_gradient_boosting_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return Gradient Boosting classifiers with different complexities."""

    configurations = [
        (20, 2),
        (50, 2),
        (20, 4),
        (50, 4),
    ]

    return {
        f"gb_{n_trees}_trees_depth_{depth}": GradientBoostingClassifier(
            n_estimators=n_trees,
            max_depth=depth,
            learning_rate=0.1,
            random_state=seed,
        )
        for n_trees, depth in configurations
    }


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted models, keyed by a unique name used in result tables.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    return {
        **get_random_forest_models(seed),
        **get_gradient_boosting_models(seed),
    }
