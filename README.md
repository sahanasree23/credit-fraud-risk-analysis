# Credit and Fraud Risk Analysis

Machine learning analysis of two public financial datasets, evaluating credit default and
transaction fraud prediction under severe class imbalance.

## Overview

This project compares logistic regression and gradient boosting on two datasets with very
different class balances, and examines how evaluation metric choice changes which model
appears better. The central finding is that predictive performance and explainability run in
opposite directions across the two sources: the fraud data supports stronger discrimination
but consists of anonymised principal components, while the credit data yields interpretable
attributes capable of justifying an adverse lending decision.

## Datasets

Neither dataset is included in this repository (see [Data setup](#data-setup)).

| | Dataset A | Dataset B |
|---|---|---|
| Name | Credit Card Fraud Detection | Default of Credit Card Clients |
| Source | [Kaggle / ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | [UCI Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Rows | 284,807 | 30,000 |
| Attributes | 31 | 24 |
| Positive rate | 0.172% | 22.12% |
| Feature type | PCA-transformed, anonymised | Demographic and behavioural |
| Licence | DbCL v1.0 | CC BY 4.0 |

## Data setup

The datasets are excluded from version control (the fraud CSV is ~144 MB, above GitHub's
100 MB file limit). To reproduce:

1. Download `creditcard.csv` from the Kaggle link above (free account required)
2. Download `default of credit card clients.xls` from the UCI link above
3. Place both in `data/`
4. Run the notebook top to bottom — Step 2 converts the `.xls` to a clean CSV

## Method

- **Models:** logistic regression (interpretable linear baseline) and histogram-based
  gradient boosting (non-linear, captures interactions)
- **Split:** 75:25 stratified, fixed random seed
- **Class imbalance:** balanced class weighting; decision thresholds tuned to maximise F2
  rather than fixed at 0.5
- **Metrics:** precision-recall AUC reported against the no-skill baseline, with recall
  prioritised over precision given asymmetric error costs

Accuracy is reported for completeness but is not used to compare models on the fraud
dataset, where predicting the negative class throughout scores 99.83% while detecting
nothing.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib xlrd openpyxl
jupyter notebook credit_fraud_risk_analysis.ipynb
```

Runtime is roughly 8 minutes, almost all in the gradient boosting step.

## Repository structure

```
.
├── credit_fraud_risk_analysis.ipynb   # main analysis
├── data/                              # datasets (not tracked)
├── figures/                           # generated plots
└── README.md
```

## References

Dal Pozzolo, A., Caelen, O., Johnson, R.A. and Bontempi, G. (2015). *Credit Card Fraud
Detection Dataset*. Machine Learning Group, Université Libre de Bruxelles.

Yeh, I-C. (2016). *Default of Credit Card Clients*. UCI Machine Learning Repository.
doi:10.24432/C55S3H

Yeh, I-C. and Lien, C-H. (2009). The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients. *Expert Systems with
Applications*, 36(2), 2473–2480.
