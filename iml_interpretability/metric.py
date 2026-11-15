"""A sample-aware interpretability metric for fitted random forests."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Interface expected by the starter repository's evaluation function."""

    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Return a numeric interpretability score for a fitted model."""


class ForestExplanationBurden(InterpretabilityMetric):
    """Measure how much work explaining individual forest predictions takes.

    The measurement combines THREE different obstacles to explaining a prediction:
      1. Number of split decisions along paths actually visited by the examples.
      2. Changes of feature between consecutive splits (context switching).
      3. Disagreement of individual trees about the predicted class.

    For sample x, let d(x) be mean number of split nodes per tree,
    s(x) be mean number of feature switches along those paths, and
    v(x) = 1 - (largest fraction of trees voting for the same class).

    Burden(x) = log2(1 + T) * [d(x) + switch_weight*s(x)]
                * [1 + disagreement_weight*v(x)]
    Interpretability = 100 / [1 + mean_x Burden(x)]

    Scores lie in (0, 100], with HIGHER scores meaning EASIER to explain.
    This is a proposed structural proxy, not a validated measure of human
    comprehension. The weights and the log2 aggregation are design choices.
    Scores are dataset-conditional and should be assessed on comparable examples.
    """

    name = "forest_explanation_score"

    def __init__(self, switch_weight: float = 0.5, disagreement_weight: float = 1.0):
        if switch_weight < 0 or disagreement_weight < 0:
            raise ValueError("Weights must not be negative")
        self.switch_weight = switch_weight
        self.disagreement_weight = disagreement_weight

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        if not isinstance(model, RandomForestClassifier):
            raise TypeError("ForestExplanationBurden requires a fitted RandomForestClassifier")
        if not hasattr(model, "estimators_"):
            raise ValueError("The random forest must be fitted before scoring")
        if len(dataset.X) == 0:
            raise ValueError("Dataset must contain at least one observation")

        X = dataset.X.to_numpy()
        n_samples = len(X)
        n_trees = len(model.estimators_)
        path_decisions = np.zeros(n_samples, dtype=float)
        feature_switches = np.zeros(n_samples, dtype=float)
        # One vote per class and tree. Use forest classes to define the vote columns.
        class_to_index = {cls: i for i, cls in enumerate(model.classes_)}
        votes = np.zeros((n_samples, len(model.classes_)), dtype=np.int32)
        sample_indices = np.arange(n_samples)

        for tree in model.estimators_:
            csr_paths = tree.decision_path(X)
            tree_features = tree.tree_.feature
            for i in range(n_samples):
                path_node_ids = csr_paths.indices[csr_paths.indptr[i] : csr_paths.indptr[i + 1]]
                # Leaves use feature=-2; retain only internal split nodes.
                features_on_path = tree_features[path_node_ids]
                features_on_path = features_on_path[features_on_path >= 0]
                path_decisions[i] += len(features_on_path)
                if len(features_on_path) > 1:
                    feature_switches[i] += np.count_nonzero(
                        features_on_path[1:] != features_on_path[:-1]
                    )

            predictions = tree.predict(X)
            for i, label in enumerate(predictions):
                votes[i, class_to_index[label]] += 1

        mean_depth = path_decisions / n_trees
        mean_switches = feature_switches / n_trees
        disagreement = 1.0 - votes.max(axis=1) / n_trees
        burden = (
            np.log2(1.0 + n_trees)
            * (mean_depth + self.switch_weight * mean_switches)
            * (1.0 + self.disagreement_weight * disagreement)
        )
        return float(100.0 / (1.0 + burden.mean()))
