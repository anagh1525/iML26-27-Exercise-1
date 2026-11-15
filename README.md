# iML 2026/27 – Exercise 1: Quantifying Interpretability

## Overview

This project investigates the relationship between **model complexity, interpretability, and predictive performance** in tree-based machine learning models.

We compare two classification algorithms:

- **Random Forest Classifier**
- **Gradient Boosting Classifier**

Both algorithms are evaluated on the **Titanic** and **Iris** datasets using a proposed interpretability metric called **Tree Explanation Burden (TEB)**.

The objective is to investigate how tree depth, ensemble size, and decision-path complexity affect the difficulty of interpreting machine learning predictions.

## 1. Research Question

**How do Random Forest and Gradient Boosting differ in predictive accuracy and structural interpretability across different datasets?**

The experiments investigate:

1. How increasing the number and depth of trees affects interpretability.
2. Whether higher predictive accuracy requires more complex models.
3. Whether Random Forest and Gradient Boosting show different accuracy–interpretability trade-offs.
4. Whether an approximate interpretability threshold can be identified.

## 2. Datasets

Two classification datasets are used.

| Dataset | Samples | Features | Classes |
|---|---:|---:|---:|
| Titanic | 1,309 | 8 processed | 2 |
| Iris | 150 | 4 | 3 |

### Titanic

The Titanic dataset is retrieved from OpenML using `load_titanic()`.

Preprocessing includes:

- Removing potentially target-leaking or unsuitable columns.
- Replacing missing numerical values using median imputation.
- Converting categorical variables to numerical representations using one-hot encoding.

**Target:** Passenger survival (0 or 1).

### Iris

The Iris dataset is loaded from scikit-learn using `load_iris()`.

It contains four numerical measurements describing three Iris flower species.

**Target:** Flower species (three classes).

## 3. Machine Learning Models

### Random Forest

Random Forest combines multiple randomized decision trees to make predictions through ensemble voting.

Eight configurations are evaluated:

| Number of Trees | Maximum Depth |
|---:|---:|
| 1 | 2 |
| 5 | 2 |
| 20 | 2 |
| 5 | 4 |
| 20 | 4 |
| 50 | 4 |
| 20 | 8 |
| 50 | 8 |

### Gradient Boosting

Gradient Boosting constructs an additive ensemble by sequentially fitting trees to improve the model's predictions.

Four configurations are evaluated:

| Boosting Stages | Maximum Depth |
|---:|---:|
| 20 | 2 |
| 50 | 2 |
| 20 | 4 |
| 50 | 4 |

The learning rate is fixed at `0.1`.

For multiclass classification, a boosting stage may contain multiple component trees.

**Total experiment configurations:** 12 per dataset, producing 24 model–dataset evaluations.

## 4. Proposed Interpretability Metric

### Tree Explanation Burden (TEB)

We use a proposed structural metric that estimates the difficulty of explaining predictions from tree-based ensembles.

The metric considers three factors:

1. **Ensemble size:** The total number of component trees.
2. **Decision-path length:** The number of split decisions followed when predicting a sample.
3. **Feature switching:** The number of times consecutive decision nodes use different input features.

For an individual sample \(x\), the explanation burden is:

\[
B(x)=\log_2(1+T)\left[d(x)+0.5s(x)\right]
\]

Where:

- \(T\) = total number of component trees.
- \(d(x)\) = average decision-path length across the trees.
- \(s(x)\) = average feature switches along those decision paths.
- \(0.5\) = proposed feature-switch penalty weight.

The final interpretability score is:

\[
I=\frac{100}{1+\frac{1}{N}\sum_{i=1}^{N}B(x_i)}
\]

Where \(N\) is the number of evaluated samples.

**A higher score represents lower estimated structural explanation burden.**

The metric goes beyond counting input features or leaves by evaluating the actual paths used for predictions and the complexity of the ensemble.

The formula and its weights are exploratory design choices, not validated measurements of human interpretability.

### Why a Common Metric?

An earlier Random Forest-specific metric included disagreement among individual trees.

Tree disagreement was removed for the cross-algorithm comparison because Gradient Boosting combines numerical contributions from trees rather than independent class votes.

TEB therefore provides a common structural scoring method for Random Forest and Gradient Boosting.

## 5. Evaluation Methodology

Each dataset is divided using a stratified train/test split:

- Training data: 75%
- Testing data: 25%
- Random seed: 0

Models are trained on the training portion and evaluated using the held-out test portion.

### Performance Metrics

- **Accuracy:** Proportion of correctly classified observations.
- **Balanced Accuracy:** Average recall across classes.
- **Macro F1-score:** Unweighted average F1-score across classes.

### Interpretability Metric

Tree Explanation Burden is calculated on the held-out input samples.

The same interpretability formula is applied to both algorithm families.

## 6. Interpretability Threshold

An important objective is to investigate when model complexity becomes too large for practical human interpretation.

We examine the calculated interpretability scores using exploratory cutoffs.

| Proposed cutoff | Interpretation |
|---:|---|
| 5 | Exploratory lower cutoff |
| 10 | Primary illustrative cutoff |
| 15 | Exploratory higher cutoff |

For the illustrative cutoff of 10:

- Score ≥ 10: At or above the proposed interpretability cutoff.
- Score below 10: Below the proposed interpretability cutoff.

**These values are heuristic thresholds, not scientifically validated boundaries of human comprehension.**

An actual threshold would require additional evidence, such as measuring human explanation accuracy and completion time for models of varying complexity.

## 7. Experimental Results

The experiment generates:

`results/scores.csv`

The results allow comparisons of:

- Random Forest versus Gradient Boosting.
- Titanic versus Iris.
- Predictive accuracy versus interpretability.
- The influence of tree count and maximum depth.
- Sensitivity to proposed interpretability cutoffs.

**The numerical results and conclusions should be added after executing the final two-dataset experiment.**

The analysis should identify the most accurate model, the model with the highest interpretability score, and whether model rankings change between datasets.

## 8. Project Structure

```text
iml_interpretability/
├── __init__.py
├── data.py
├── models.py
├── metric.py
└── evaluate.py

scripts/
└── run_experiment.py

tests/
└── test_smoke.py

results/
└── scores.csv

presentation/
└── presentation.pdf

README.md
pyproject.toml
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `data.py` | Load and preprocess Titanic and Iris |
| `models.py` | Define Random Forest and Gradient Boosting configurations |
| `metric.py` | Implement Tree Explanation Burden |
| `evaluate.py` | Train models and calculate evaluation scores |
| `__init__.py` | Export package functions and classes |
| `run_experiment.py` | Execute the experiments and save results |
| `tests/` | Verify basic implementation correctness |

## 9. Installation and Reproduction

### Using Python venv and pip

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

### Run Tests

```bash
python -m pytest
```

### Run Experiments

```bash
python scripts/run_experiment.py
```

The script evaluates both algorithms on Titanic and Iris and writes the results to:

```text
results/scores.csv
```

The Titanic dataset is downloaded from OpenML on the first run and cached locally. Iris is provided by scikit-learn.

### Alternative: Using uv

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py
```

## 10. Limitations and Future Work

The proposed interpretability score has several limitations:

- It measures structural complexity rather than human understanding directly.
- The feature-switch penalty and ensemble-size adjustment are heuristic.
- Random Forest and Gradient Boosting combine component trees differently.
- The number of trees alone may not accurately represent explanation difficulty.
- Only two datasets are evaluated.
- A single train/test split does not capture uncertainty across different random seeds.
- Proposed interpretability thresholds have not been validated with human participants.

Future work could investigate human comprehension, explanation time, alternative metric weights, repeated experimental splits, and additional datasets.

## 11. AI Statement

ChatGPT (OpenAI) was used to help understand the assignment requirements, discuss interpretability metric design, assist with Python implementation and debugging, explain evaluation results, and draft project documentation and presentation content.

AI assistance was also used when extending the implementation from Random Forest on Titanic to include Gradient Boosting and Iris. GitHub Copilot was consulted during the development process.

The code was tested locally using pytest, and the original Titanic Random Forest experiment was executed locally. The final extended experiment, its results, and the interpretability conclusions should be independently checked before submission.

The project author is responsible for reviewing the implementation, understanding the metric, verifying the outputs, and ensuring the submitted results are accurate.

## 12. Presentation

The five-minute presentation is located at:

`presentation/presentation.pdf`

It summarizes the research motivation, proposed metric, experimental setup, model comparisons, findings, limitations, and AI usage.

## Repository

https://github.com/anagh1525/iML26-27-Exercise-1
