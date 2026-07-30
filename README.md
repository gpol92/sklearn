This repository contains a script CLI in Python that allows to explore the allowed and disallowed combinations of parameters of the LogisticRegression model of Scikit-learn


Example of usage

uv run python ml_cli.py --logistic 8 4000 lbfgs --intercept

Example of output

Logistic regression chosen
Training Logistic Regression
rs=8
max_iter=4000
solver=lbfgs
dual=False
intercept=True
Logistic Regression model accuracy: 96.49%
