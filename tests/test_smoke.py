import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from iml_interpretability.data import Dataset
from iml_interpretability.metric import TreeExplanationBurden


def test_random_forest_metric():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(100, 4)),
        columns=["a", "b", "c", "d"],
    )

    y = pd.Series((X["a"] + X["b"] > 0).astype(int))

    dataset = Dataset(name="synthetic", X=X, y=y)

    model = RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y)

    metric = TreeExplanationBurden()
    score = metric.score(model, dataset)

    assert np.isfinite(score)
    assert 0 <= score <= 100