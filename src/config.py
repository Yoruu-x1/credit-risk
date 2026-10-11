"""Pengaturan dataset, schema awal, split, dan biaya bisnis."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

# Semua lokasi file merujuk ke folder proyek ini.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

DATA_PATH: Path = DATA_DIR / "UCI_Credit_Card.csv"

CSV_SEPARATOR: str = ","
CSV_ENCODING: str = "utf-8-sig"

# Nama target pada versi CSV yang umum digunakan.
# Jika header filemu berbeda, ubah hanya pengaturan ini.
RAW_TARGET_COLUMN: str = "default.payment.next.month"
TARGET_NAME: str = "default_next_month"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

# Asumsi biaya relatif, bukan nominal kerugian uang aktual.
FALSE_NEGATIVE_COST: float = 5.0
FALSE_POSITIVE_COST: float = 1.0

# ID nasabah dan indeks ekspor CSV tidak menjadi fitur model.
EXCLUDED_COLUMNS: tuple[str, ...] = ("ID", "Unnamed: 0")

DEMOGRAPHIC_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "SEX",
    "EDUCATION",
    "MARRIAGE",
)

REPAYMENT_STATUS_COLUMNS: tuple[str, ...] = (
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
)

BILL_COLUMNS: tuple[str, ...] = tuple(
    f"BILL_AMT{month}" for month in range(1, 7)
)

PAYMENT_COLUMNS: tuple[str, ...] = tuple(
    f"PAY_AMT{month}" for month in range(1, 7)
)

# Kandidat kategori untuk EDA.
# Perlakuan repayment status pada model ditentukan setelah eksplorasi.
EDA_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    *DEMOGRAPHIC_CATEGORICAL_COLUMNS,
    *REPAYMENT_STATUS_COLUMNS,
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "LIMIT_BAL",
    "AGE",
    *BILL_COLUMNS,
    *PAYMENT_COLUMNS,
)

FEATURE_COLUMNS: tuple[str, ...] = (
    *NUMERIC_COLUMNS,
    *EDA_CATEGORICAL_COLUMNS,
)

# Mapping hanya untuk membaca output EDA, belum mengubah fitur model.
# Kode yang belum tercantum akan tetap ditampilkan apa adanya.
CATEGORY_LABELS: dict[str, dict[int, str]] = {
    "SEX": {
        1: "Male",
        2: "Female",
    },
   "EDUCATION": {
    0: "Undocumented education code 0",
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others",
    5: "Unknown (code 5)",
    6: "Unknown (code 6)",
    },
    "MARRIAGE": {
        1: "Married",
        2: "Single",
        3: "Others",
    },
}

# Pengaturan CV yang sama untuk perbandingan baseline.
CV_FOLDS: int = 5
CV_RANDOM_STATE: int = 42

# Keputusan baseline: semua kode kategori dipertahankan melalui OHE.
CATEGORICAL_COLUMNS: tuple[str, ...] = EDA_CATEGORICAL_COLUMNS

LR_BASELINE_PARAMS: dict[str, float | int | str] = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 5000,
    "tol": 1e-4,
    "random_state": RANDOM_STATE,
}

COST_RATIOS: tuple[float, ...] = (2.0, 5.0, 10.0)

EXPERIMENTS_DIR: Path = REPORTS_DIR / "experiments"

EXPERIMENT_FILES: dict[str, str] = {
    "summary": "summary.csv",
    "fold_metrics": "fold_metrics.csv",
    "oof_predictions": "oof_predictions.csv",
    "threshold_curve": "threshold_curve.csv",
    "feature_mapping": "feature_mapping_first_fold.csv",
    "metadata": "metadata.json",
    "cost_sensitivity": "cost_sensitivity.csv",
}

LGBM_BASELINE_PARAMS: dict[str, float | int | str | bool] = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "random_state": RANDOM_STATE,
    "n_jobs": 4,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

XGB_BASELINE_PARAMS: dict[str, str | int | float] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cpu",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": 4,
    "verbosity": 1,
}

ENGINEERED_NUMERIC_COLUMNS: tuple[str, ...] = (
    "all_amounts_zero",
    "zero_payment_months",
    "delayed_months_count",
)

ENGINEERED_FEATURE_DEFINITIONS: dict[str, str] = {
    "all_amounts_zero": (
        "1 if all six BILL_AMT and all six PAY_AMT values are zero; "
        "otherwise 0"
    ),
    "zero_payment_months": (
        "Number of PAY_AMT1-PAY_AMT6 values equal to zero"
    ),
    "delayed_months_count": (
        "Number of PAY_0, PAY_2-PAY_6 codes greater than zero"
    ),
}
FE_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "payment_mean",
    "payment_std",
    "bill_mean",
    "bill_std",
)

FE_RELATIVE_COLUMNS: tuple[str, ...] = (
    "bill_latest_to_limit",
    "payment_mean_to_limit",
)

FE_TEMPORAL_COLUMNS: tuple[str, ...] = (
    "payment_recent_minus_old",
    "bill_recent_minus_old",
    "max_delay",
)

AVAILABLE_ENGINEERED_COLUMNS: tuple[str, ...] = (
    *ENGINEERED_NUMERIC_COLUMNS,
    *FE_AGGREGATE_COLUMNS,
    *FE_RELATIVE_COLUMNS,
    *FE_TEMPORAL_COLUMNS,
)

ENGINEERED_FEATURE_DEFINITIONS.update(
    {
        "payment_mean": "Mean of PAY_AMT1-PAY_AMT6",
        "payment_std": (
            "Population standard deviation of PAY_AMT1-PAY_AMT6"
        ),
        "bill_mean": "Mean of BILL_AMT1-BILL_AMT6",
        "bill_std": (
            "Population standard deviation of BILL_AMT1-BILL_AMT6"
        ),
        "bill_latest_to_limit": "BILL_AMT1 divided by LIMIT_BAL",
        "payment_mean_to_limit": (
            "Mean of PAY_AMT1-PAY_AMT6 divided by LIMIT_BAL"
        ),
        "payment_recent_minus_old": (
            "Mean of PAY_AMT1-PAY_AMT3 minus mean of PAY_AMT4-PAY_AMT6"
        ),
        "bill_recent_minus_old": (
            "Mean of BILL_AMT1-BILL_AMT3 minus mean of BILL_AMT4-BILL_AMT6"
        ),
        "max_delay": (
            "Maximum positive repayment-status code; "
            "zero if no positive code exists"
        ),
    }
)

# Fitur untuk eksperimen refinement setelah error analysis.
FE_REFINEMENT_RATIO_COLUMNS: tuple[str, ...] = (
    *FE_RELATIVE_COLUMNS,
    "bill_change_to_limit",
)

FE_REFINEMENT_BEHAVIOR_COLUMNS: tuple[str, ...] = (
    "all_amounts_zero",
    "zero_payment_months",
)

AVAILABLE_ENGINEERED_COLUMNS = (
    *AVAILABLE_ENGINEERED_COLUMNS,
    "bill_change_to_limit",
)

ENGINEERED_FEATURE_DEFINITIONS["bill_change_to_limit"] = (
    "(Mean of BILL_AMT1-BILL_AMT3 minus mean of "
    "BILL_AMT4-BILL_AMT6) divided by LIMIT_BAL"
)

# Artefak final dan pengaturan inference.
FINAL_MODEL_NAME: str = "blend_refinement_xgb75_v09"
FINAL_MODEL_DIR: Path = MODELS_DIR / FINAL_MODEL_NAME
FINAL_MODEL_PATH: Path = FINAL_MODEL_DIR / "model.joblib"
FINAL_METADATA_PATH: Path = FINAL_MODEL_DIR / "metadata.json"

SHAP_BACKGROUND_PATH: Path = FINAL_MODEL_DIR / "background.csv"
SHAP_METADATA_PATH: Path = FINAL_MODEL_DIR / "explanation_metadata.json"
FINAL_SCHEMA_PATH: Path = FINAL_MODEL_DIR / "categorical_schema.json"

SHAP_REPORTS_DIR: Path = REPORTS_DIR / "explainability" / FINAL_MODEL_NAME
INFERENCE_CONSTRAINTS_PATH: Path = FINAL_MODEL_DIR / "runtime-constraints.txt"

SHAP_BACKGROUND_SIZE: int = 100
SHAP_SAMPLE_SIZE: int = 128
SHAP_ADDITIVITY_ATOL: float = 1e-5
API_TOP_REASONS: int = 5