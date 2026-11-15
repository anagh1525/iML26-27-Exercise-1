from pathlib import Path

from iml_interpretability import (
    load_titanic,
    load_iris,
    get_models,
    evaluate,
    TreeExplanationBurden,
)

SEED = 0
OUTPUT = Path("results/scores.csv")


def main():
    results = evaluate(
        datasets=[
            load_titanic(),
            load_iris(),
        ],
        models=get_models(seed=SEED),
        metrics=[TreeExplanationBurden()],
        seed=SEED,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(OUTPUT, index=False)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()