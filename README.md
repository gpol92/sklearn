# sklearn CLI

This repository contains a small Python CLI script (ml_cli.py) to explore allowed and disallowed combinations of parameters for scikit-learn's LogisticRegression model.

The script accepts arguments to configure a LogisticRegression run and prints the chosen configuration and resulting model accuracy.

## Requirements

- Python 3.8+
- scikit-learn

Install dependencies with pip if needed, for example:

```bash
pip install scikit-learn
```

## Usage

Run the CLI and pass the `--logistic` flag followed by the random state, max iterations, solver name, and any optional flags like `--intercept`.

Example:

```bash
python ml_cli.py --logistic 8 4000 lbfgs --intercept
```

## Examples

Below are example runs and their example output.

Example 1

```bash
python ml_cli.py --logistic 8 4000 lbfgs --intercept
```

Example output:

```
Logistic regression chosen
Training Logistic Regression
rs=8
max_iter=4000
solver=lbfgs
dual=False
intercept=True
Logistic Regression model accuracy: 96.49%
```

Example 2

```bash
python ml_cli.py --logistic 8 4000 newton-cholesky --intercept
```

Example output:

```
Logistic regression chosen
Training Logistic Regression
rs=8
max_iter=4000
solver=newton-cholesky
dual=False
intercept=True
Logistic Regression model accuracy: 96.49%
```

## Notes

- The purpose of this CLI is exploratory: to test different parameter combinations for scikit-learn's LogisticRegression and observe which combinations are valid and how they affect results.
- If you encounter errors with a particular solver or parameter combination, check scikit-learn's LogisticRegression documentation for solver-specific requirements (for example, some solvers do not support the `dual` parameter or certain penalty types).

## License

See repository for license information.
