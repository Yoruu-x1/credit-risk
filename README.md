# Credit Card Default Risk API

An end-to-end machine learning project for predicting next-month credit
card default, with cost-sensitive threshold selection, SHAP explanations,
and a Dockerized FastAPI service.

## Problem and decision policy

Missing a defaulting customer is assigned five times the cost of falsely
flagging a non-defaulting customer:

**Cost = 5 × False Negatives + False Positives**

These are assumed relative cost units, not measured monetary losses.
Class 1 represents default.

The API returns a default probability and a risk flag. It does not
automatically approve or reject credit applications.

## Dataset

This project uses the UCI **Default of Credit Card Clients** dataset:
30,000 customers and 23 input features covering credit limits,
demographics, repayment status, bills, and payments.

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Citation: Yeh, I. (2009). Default of Credit Card Clients.
UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

Dataset license: CC BY 4.0.
The SHAP background file contains a subset of dataset features.

## Experiment workflow

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Data exploration and quality checks |
| `02_baseline_models.ipynb` | Baseline model comparison |
| `03_optimization.ipynb` | Hyperparameter optimization |
| `04_ensembles.ipynb` | Ensemble experiments |
| `05_model_refinement.ipynb` | Error analysis, feature and representation experiments |
| `06_final_evaluation.ipynb` | Locked-model evaluation on the held-out test set |
| `07_shap.ipynb` | Global and individual prediction explanations |

The stratified split reserves 24,000 development rows and 6,000 test rows.
Preprocessing is fitted within training folds.

Development out-of-fold predictions are used for model comparison and
cost-based threshold selection. The final model and threshold were locked
before test evaluation.

The test set was evaluated once and was not used to select the model,
ensemble weights, or threshold.

Notebooks document the experiment history. Running the API does not
require rerunning the notebooks.

## Selected model

**Weighted soft voting: 75% XGBoost + 25% LightGBM**

- Artifact identifier: `blend_refinement_xgb75_v09`
- Locked threshold: `0.1589311876981523`
- Each member includes its own feature engineering and preprocessing.
- Final artifacts are included under `models/blend_refinement_xgb75_v09/`.

The ensemble had the lowest observed development cost among the reported
candidates. Its improvement over the refined standalone XGBoost model was
small; this does not establish a statistically significant advantage.

## Held-out test results

| Metric | Result |
|---|---:|
| Test rows | 6,000 |
| ROC-AUC | 0.78332 |
| Average precision | 0.55913 |
| F1 | 0.50173 |
| Recall | 76.41% |
| Precision | 37.35% |
| Flagged customers | 45.25% |
| Brier score | 0.13472 |
| Log loss | 0.42825 |
| Cost per 1,000 customers | 544.33 |

Confusion matrix counts: TN = 2,972; FP = 1,701; FN = 313; TP = 1,014.

| Policy | Relative cost per 1,000 |
|---|---:|
| Predict non-default for everyone | 1,105.83 |
| Flag everyone as high risk | 778.83 |
| Selected model and threshold | 544.33 |

## Run with Docker

Install Docker and start its engine. From the repository root:

```bash
docker build -t credit-default-api:0.1 .
docker run --rm --name credit-default-api -p 127.0.0.1:8000:8000 credit-default-api:0.1
```

Open:

- Health: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

Use the example request in Swagger to call:

- `POST /predict?explain=false`
- `POST /predict?explain=true`

The container includes the final model and does not require the raw dataset.

## SHAP explanations

With `explain=true`, the response includes a reference probability,
the five largest absolute feature contributions, and the combined
contribution of the remaining features.

Contributions are reported in percentage points. Positive values increase
the model's predicted default probability; negative values decrease it
relative to the reference probability.

One-hot encoded contributions are grouped under their original feature
names. Derived features are reported separately. Member explanations are
combined using the same weights as the prediction ensemble.

The API checks that the reference probability plus all contributions
reconstructs the predicted probability within numerical tolerance.

SHAP explains model behavior; it does not establish causation.

## Tests

In a Python 3.11 virtual environment:

```bash
python -m pip install -r requirements-runtime.txt
python -m pip install -c requirements.txt pytest httpx
python -m pytest tests/test_api.py -q
```

The API tests cover prediction consistency, invalid inputs,
categorical schema definitions, and explanation reconstruction.

Local validation: **7 tests passed**.
Docker smoke checks verified health, prediction, and SHAP for a sample request.

## Limitations

- Historical Taiwanese data may not represent present-day customers or
  other markets.
- The random holdout does not establish performance under future temporal drift.
- The 5:1 cost ratio is an assumption and requires business validation.
- Precision is 37.35% at the selected operating point, implying many
  false positive flags.
- Fairness, load capacity, and operational monitoring have not been validated.
- This is a portfolio implementation, not a validated lending system.