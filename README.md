# Historical Company Risk Underwriting MVP

A minimal end-to-end machine-learning application that estimates a company's
one-year bankruptcy probability from historical financial ratios.

Bankruptcy is used here as an observable proxy for severe company risk. That
makes the task useful for demonstrating commercial-insurance underwriting
triage, where a model can help reviewers prioritize the highest-risk cases. It
is not a production underwriting or pricing system.

## Dataset

The project uses the UCI Polish Companies Bankruptcy dataset. The first MVP
trains only on `data/raw/5year.arff`: financial ratios from the final observed
year are used to predict bankruptcy one year later.

The file contains 64 numerical ratios (`X1` to `X64`) and a binary bankruptcy
label. The loader converts `?` and infinite values to missing values, then the
training workflow fits median imputation using only the training split.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure the dataset is available at:

```text
data/raw/5year.arff
```

## Train

Run the complete data preparation, training, calibration, and evaluation
workflow:

```bash
python -m src.train
```

The workflow uses a stratified 80/20 train/test split with random seed 42,
median imputation, imbalance-aware CatBoost, and three-fold probability
calibration. Isotonic calibration is attempted first, with sigmoid calibration
as a fallback.

CatBoost does not require feature scaling for this data, so `scaler.joblib`
intentionally stores `None` while preserving a stable artifact interface.

## Results

The trained model was evaluated on the held-out test set of 1,182 companies,
including 82 bankruptcies. The saved results in `models/metrics.json` are:

| Metric | Result |
| --- | ---: |
| ROC-AUC | 0.956 |
| PR-AUC / average precision | 0.728 |
| Brier score | 0.0332 |
| Accuracy at probability threshold 0.5 | 95.4% |
| Precision at threshold 0.5 | 79.2% |
| Recall at threshold 0.5 | 46.3% |
| F1 at threshold 0.5 | 0.585 |
| Recall among the top 10% highest-risk companies | 78.0% |
| Bankruptcy rate in the top 10% risk group | 53.8% |
| Baseline test-set bankruptcy rate | 6.9% |

The ROC-AUC of 0.956 indicates that the model ranks bankrupt companies above
non-bankrupt companies very effectively. The PR-AUC of 0.728 is particularly
useful here because bankruptcy is rare: it shows that the model retains strong
precision and recall despite the imbalanced target.

At the conventional 0.5 probability threshold, the model correctly identified
38 of 82 bankruptcies and produced 10 false positives. This gives high
precision, but recall is only 46.3%, so a 0.5 threshold is too conservative if
the underwriting goal is to find most risky companies.

The top-decile result is more relevant for underwriting triage. Reviewing the
119 companies with the highest predicted risk would have found 64 of the 82
bankruptcies, or 78.0%, while reviewing only about 10% of the test portfolio.
The bankruptcy rate in this group was 53.8%, compared with 6.9% across the
whole test set. In other words, the selected group was about 7.8 times as
concentrated with bankruptcies as the baseline portfolio.

The Brier score of 0.0332 measures probability error, where lower is better.
The model used isotonic calibration to make its probabilities more meaningful,
not just its ranking. Calibration performance can be inspected in
`reports/calibration_curve.csv` and in the Streamlit performance view.

These are promising prototype results, but they come from one random holdout
of a historical dataset. They should not be interpreted as evidence that the
same performance would hold for current companies, another country, or actual
insurance claims.

## Run The App

```bash
streamlit run app/streamlit_app.py
```

The app supports:

- inspecting a held-out company, its true outcome, risk score, percentile, and
  important financial ratios;
- editing the eight most important ratios on top of an existing company
  template;
- reviewing discrimination, calibration, top-decile triage metrics, a
  confusion matrix, and global feature importance.

## Outputs

Training creates:

```text
data/processed/train.csv
data/processed/test.csv
models/model.joblib
models/calibrated_model.joblib
models/imputer.joblib
models/scaler.joblib
models/feature_names.joblib
models/metrics.json
reports/feature_importance.csv
reports/test_predictions.csv
reports/calibration_curve.csv
```

`test_predictions.csv` includes calibrated probability, risk percentile,
the threshold-0.5 decision, and exact top-10%-risk triage membership.
`feature_importance.csv` contains CatBoost global importance with readable
ratio descriptions.

## Limitations

- Bankruptcy is only a proxy for insurance loss and does not model claim
  frequency, severity, coverage, pricing, or exposure.
- The data is historical and country-specific, so performance may not transfer
  to current companies or other markets.
- The test split is a demonstration holdout, not a temporal or external
  validation set.
- Median comparisons are simple explanations, not causal claims or local SHAP
  attributions.
- Risk buckets are illustrative and are not validated underwriting rules.
- The model must not be used as the sole basis for real underwriting decisions.

The next version could train across all ARFF files and include the forecasting
horizon as an input feature.
