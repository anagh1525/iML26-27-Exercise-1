"""Fit random forests and evaluate test accuracy and interpretation burden."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from iml_interpretability.data import Dataset
from iml_interpretability.metric import InterpretabilityMetric


def evaluate(
    datasets: Iterable[Dataset],
    models: dict[str, ClassifierMixin],
    metrics: Iterable[InterpretabilityMetric],
    seed: int = 0,
    test_size: float = 0.25,
) -> pd.DataFrame:
    """Fit each model on each dataset using one stratified train/test split.

    Accuracy is measured on the held-out set; the interpretability metric is
    computed on that same held-out feature matrix to reflect explanation burden
    for unseen predictions. No test labels are used by the interpretability metric.
    """
    metrics = list(metrics)
    rows: list[dict] = []
    for dataset in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=test_size, random_state=seed,
            stratify=dataset.y,
        )
        test_data = Dataset(name=dataset.name, X=X_test, y=y_test)
        for model_name, model in models.items():
            fitted = clone(model).fit(X_train, y_train)
            row = {
                "dataset": dataset.name,
                "model": model_name,
                "n_samples": len(dataset.X),
                "n_features": dataset.X.shape[1],
                "n_trees": fitted.n_estimators,
                "max_depth_setting": fitted.max_depth,
                "accuracy": float(accuracy_score(y_test, fitted.predict(X_test))),
            }
            for metric in metrics:
                row[metric.name] = metric.score(fitted, test_data)
            rows.append(row)
    return pd.DataFrame(rows)
