
"""Interpretable Machine Learning - Exercise 1.

Package for evaluating the predictive performance and interpretability
of Random Forest and Gradient Boosting classifiers on Titanic and Iris.
"""

from iml_interpretability.data import (
    Dataset,
    load_titanic,
    load_iris,
)

from iml_interpretability.evaluate import evaluate

from iml_interpretability.metric import (
    InterpretabilityMetric,
    TreeExplanationBurden,
)

from iml_interpretability.models import (
    get_models,
    get_random_forest_models,
    get_gradient_boosting_models,
)

__all__ = [
    "Dataset",
    "load_titanic",
    "load_iris",
    "evaluate",
    "InterpretabilityMetric",
    "TreeExplanationBurden",
    "get_models",
    "get_random_forest_models",
    "get_gradient_boosting_models",
]
