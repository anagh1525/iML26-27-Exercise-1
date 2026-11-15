# Quantifying Random Forest Interpretability on Titanic

## Research question
How does the complexity of a random forest (tree count and maximum tree depth) influence the effort required to explain its predictions, and how does this compare with predictive accuracy **on the Titanic dataset**?

## Dataset
The experiment uses **only Titanic** from the original exercise template (`load_titanic()` in `iml_interpretability/data.py`). The loader fetches OpenML Titanic, version 1, and caches it under `data/`. It removes potentially leaky columns such as `boat` and `body`, imputes missing numeric values, and one-hot encodes categorical variables. The initial download requires internet access.

## Proposed interpretability metric
The **Forest Explanation Burden** metric combines three properties observed on prediction paths:

1. Average number of split decisions per tree, `d(x)`.
2. Average number of changes between consecutive splitting features, `s(x)`.
3. Tree vote disagreement, `v(x) = 1 - (largest class vote count / number of trees)`.

For `T` trees, a prediction's proposed burden is:

```
burden(x) = log2(1 + T) * (d(x) + 0.5*s(x)) * (1 + v(x))
score = 100 / (1 + mean_x burden(x))
```

Larger scores indicate **lower estimated explanation burden**. The metric is an **original, exploratory structural proxy** and does not demonstrate that a particular score is a human-validated interpretability threshold. The weights and logarithmic tree-count term are design choices to review critically. In particular, disagreement might not faithfully track the effort of explaining a forest, and all-tree explanations may be impractical.

## Experimental design
Eight `RandomForestClassifier` configurations vary tree count (`1`, `5`, `20`, `50`) and maximum tree depth (`2`, `4`, `8`) in a controlled selection. All eight are trained on the **same Titanic training split**. Accuracy and interpretability are measured on the same held-out 25% of the Titanic data, with a stratified train/test split and `random_state=0`.

The resulting eight rows show how interpretable each forest is under the proposed metric and how accurately it predicts survival on held-out Titanic examples.

## Reproduce
Copy the files into your fork of the provided course repository and, from its root, run:

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py
```

The experiment writes `results/scores.csv`. **Only Titanic is loaded by the experiment.** Tests use a small synthetic, Titanic-shaped fixture solely for offline unit testing; it is **not an experimental dataset**.

If the OpenML download fails, restore internet connectivity and retry the experiment. Do **not** present synthetic test-fixture results as experimental Titanic results.

## Interpretation and limitations
Interpretability can decrease when increasing tree count or depth, but this metric only estimates explanation effort, not human understanding. A defensible *human* threshold would require additional justification, such as a study of users' ability to follow explanations at differing complexity levels. The single-dataset experiment cannot investigate associations between dataset meta-features and interpretability (Task 2).

## AI statement
ChatGPT (OpenAI, GPT-6) assisted with the Random Forest implementation, proposed metric, tests, and README draft. **Before submission, review the code and findings yourself, edit this statement to describe what you personally verified, and take responsibility for your submission.**

## Presentation
Add `presentation/presentation.pdf` containing the motivation, metric, actual Titanic experiment results, caveats, and a slide describing AI usage. Do not insert results from other datasets.
