
"""Interpretability metrics for tree-based classification models.

This module provides a structural explanation burden metric that works
with Random Forest and Gradient Boosting classifiers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for measuring model interpretability."""

    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score for a fitted model."""


class TreeExplanationBurden(InterpretabilityMetric):
    """Measure structural explanation burden of tree ensembles.

    The score considers:
    1. Number of component trees.
    2. Average decision-path length.
    3. Average feature switching along decision paths.

    Higher scores indicate lower estimated explanation burden.
    """

    name = "tree_explanation_score"

    def __init__(self, switch_weight: float = 0.5):
        self.switch_weight = switch_weight

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Calculate interpretability score from 0 to 100."""

        if isinstance(model, RandomForestClassifier):
            trees = list(model.estimators_)

        elif isinstance(model, GradientBoostingClassifier):
            # Gradient Boosting stores trees in a 2D array.
            # Multiclass classification has multiple trees per stage.
            trees = list(model.estimators_.ravel())

        else:
            raise TypeError(
                f"Unsupported model type: {type(model).__name__}"
            )

        X = dataset.X
        n_samples = len(X)
        n_trees = len(trees)

        if n_samples == 0 or n_trees == 0:
            raise ValueError("Dataset and tree ensemble must not be empty.")

        path_depth = np.zeros(n_samples)
        feature_switches = np.zeros(n_samples)

        for tree in trees:
            decision_paths = tree.decision_path(X)

            for i in range(n_samples):
                start = decision_paths.indptr[i]
                end = decision_paths.indptr[i + 1]

                node_indices = decision_paths.indices[start:end]

                # Feature index -2 represents a leaf node.
                features = tree.tree_.feature[node_indices]
                features = features[features >= 0]

                path_depth[i] += len(features)

                if len(features) > 1:
                    feature_switches[i] += np.count_nonzero(
                        features[1:] != features[:-1]
                    )

        mean_depth = path_depth / n_trees
        mean_switches = feature_switches / n_trees

        burden = (
            np.log2(1 + n_trees)
            * (mean_depth + self.switch_weight * mean_switches)
        )

        return float(100.0 / (1.0 + np.mean(burden)))
